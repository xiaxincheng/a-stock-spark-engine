"""L1 Contrastive Reasoning — natural-law-driven strategy generation."""
import random
import json
import re
import hashlib
import os
import numpy as np
import pandas as pd
from spark_core.types import Hypothesis, Direction, HypothesisStatus
from spark_core.llm_client import llm_reason, llm_is_available


# ═══════════════════════════════════════════════════════════════
# Expression Compiler — validates and compiles LLM-generated expressions
# ═══════════════════════════════════════════════════════════════

# Available science domains for LLM to pick from
NATURAL_SCIENCE_DOMAINS = [
    "物理学", "化学", "生物学", "天文学", "地质学",
    "医学/生理学", "数学", "生态学", "流体力学", "热力学",
    "电磁学", "量子力学", "遗传学", "神经科学", "气象学",
]

from spark_core.indicator_registry import (  # noqa: E402
    FUNCTION_REGISTRY,
    PATTERN_FUNCTIONS,
    indicator_catalog_text,
)


class ExpressionCompiler:
    """Validates and compiles LLM-generated condition expressions.

    Expressions use function-call syntax like:
        close > MA(60) AND volume_ratio(5) > 1.2

    The compiler:
    1. Validates function names and parameter ranges
    2. On evaluation, pre-computes all referenced indicator columns on the DataFrame
    3. Translates the expression to pandas-eval-compatible syntax
    4. Returns a boolean mask
    """

    # Regex to match function calls: MA(60), volume_ratio(5), etc.
    FUNC_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\(([^)]*)\)")

    def __init__(self):
        self._compile_errors: list[dict] = []

    def validate(self, expr: str) -> tuple:
        """Validate an expression without executing it.

        Returns (is_valid, error_message).
        Checks: function names exist, parameter counts/ranges are correct,
        expression has comparison operators, complexity is reasonable.
        """
        if not expr or not expr.strip():
            return False, "empty expression"

        # Check for comparison operators
        if not re.search(r"[><=!]=?|==|!=", expr):
            return False, "no comparison operator found"

        # Check complexity
        and_or_count = len(re.findall(r"\b(AND|OR)\b", expr, re.IGNORECASE))
        if and_or_count > 10:
            return False, f"too many AND/OR connectors: {and_or_count} (max 10)"

        paren_depth = 0
        max_depth = 0
        for ch in expr:
            if ch == "(":
                paren_depth += 1
                max_depth = max(max_depth, paren_depth)
            elif ch == ")":
                paren_depth -= 1
        if max_depth > 4:
            return False, f"too many nested parentheses: depth {max_depth} (max 4)"

        # Validate each function call
        for match in self.FUNC_CALL_RE.finditer(expr):
            func_name = match.group(1)
            args_str = match.group(2).strip()

            # Check if it's a pattern function (no args expected)
            if func_name in PATTERN_FUNCTIONS:
                if args_str:
                    return False, f"pattern function '{func_name}' takes no arguments, got '{args_str}'"
                continue

            # Check if it's a registered function
            if func_name not in FUNCTION_REGISTRY:
                return False, f"unknown function: '{func_name}'"

            param_rules, _ = FUNCTION_REGISTRY[func_name]
            args = [a.strip() for a in args_str.split(",")] if args_str else []

            if len(args) != len(param_rules):
                return False, (
                    f"function '{func_name}' expects {len(param_rules)} params, "
                    f"got {len(args)}: {args_str}"
                )

            for i, ((pname, pmin, pmax), arg) in enumerate(zip(param_rules, args)):
                try:
                    val = float(arg)
                except ValueError:
                    return False, f"function '{func_name}' param {i+1} ('{arg}') is not numeric"
                if val < pmin or val > pmax:
                    return False, (
                        f"function '{func_name}' param '{pname}'={val} "
                        f"out of range [{pmin}, {pmax}]"
                    )

        return True, ""

    def compile_and_eval(self, expr: str, df: pd.DataFrame) -> pd.Series:
        """Compile an expression and evaluate it on a DataFrame.

        Pre-computes all referenced indicator columns, then translates
        the expression to pandas-query-compatible syntax and evaluates.
        Returns a boolean Series (same index as df).
        """
        if df.empty:
            return pd.Series(False, index=df.index)

        result_expr = expr
        computed_cols: set[str] = set()

        # Process each function call: compute column, replace in expression
        # Reverse iteration to preserve string positions during replacement
        matches = list(self.FUNC_CALL_RE.finditer(expr))
        for match in reversed(matches):
            func_name = match.group(1)
            args_str = match.group(2).strip()
            full_match = match.group(0)

            # Generate a unique column name for this function result
            col_key = f"_lc_{func_name}"
            if args_str:
                col_key += "_" + args_str.replace(",", "_").replace(".", "p").replace(" ", "")
            # Hash long column names
            if len(col_key) > 50:
                col_key = f"_lc_{func_name}_{hashlib.md5(args_str.encode()).hexdigest()[:8]}"

            if col_key not in computed_cols:
                if func_name in PATTERN_FUNCTIONS:
                    df[col_key] = PATTERN_FUNCTIONS[func_name](df)
                elif func_name in FUNCTION_REGISTRY:
                    args = [float(a.strip()) for a in args_str.split(",")] if args_str else []
                    _, computer = FUNCTION_REGISTRY[func_name]
                    df[col_key] = computer(df, *args)
                else:
                    raise ValueError(f"Unknown function: {func_name}")
                computed_cols.add(col_key)

            result_expr = result_expr.replace(full_match, col_key, 1)

        # Replace logical operators for pandas eval
        result_expr = re.sub(r"\bAND\b", "and", result_expr)
        result_expr = re.sub(r"\bOR\b", "or", result_expr)
        result_expr = re.sub(r"\bNOT\b", "not", result_expr)

        try:
            mask = df.eval(result_expr)
            return mask.astype(bool).fillna(False)
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}\nTranslated expr: {result_expr}") from e

    def make_scan_function(self, entry_expr: str, exit_expr: str | None):
        """Create a scan function compatible with STRATEGY_REGISTRY.

        Returns a callable that takes a CSV filepath and returns a list of
        signal dicts with keys: code, signal_date, close.
        """
        compiler = self

        def scan_func(fpath: str) -> list[dict]:
            try:
                # Spark signals must use the same forward-adjusted loader as
                # the production backtest; raw CSV bars would distort MA/MACD.
                from backtest_framework import load_stock_data
                df = load_stock_data(fpath)
            except Exception:
                return []

            if df is None:
                return []

            # Evaluate entry expression
            code = os.path.basename(fpath).replace(".csv", "")
            try:
                entry_mask = compiler.compile_and_eval(entry_expr, df)
            except Exception:
                return []

            signals = []
            for idx in df[entry_mask].index:
                sig = {
                    "code": code,
                    "signal_date": str(df.loc[idx, "date"]),
                    "close": float(df.loc[idx, "close"]),
                    "source": "L1_natural_law",
                }
                if exit_expr:
                    sig["exit_condition"] = exit_expr
                signals.append(sig)

            return signals

        return scan_func


# ═══════════════════════════════════════════════════════════════
# Dynamic strategy registry (module-level, shared across spark core)
# ═══════════════════════════════════════════════════════════════

DYNAMIC_STRATEGIES: dict[str, tuple] = {}
# name -> (scan_func, pf, boost)


# ═══════════════════════════════════════════════════════════════
# LLM Prompt Templates
# ═══════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = r"""
═══════════════════════════════════════════════════════════
示范1：渗透压（化学）→ 乖离率回归（均值回归型）
═══════════════════════════════════════════════════════════

科学概念：
  domain: 化学
  name: 渗透压·半透膜平衡
  mechanism: 溶质浓度差驱动水分子从低浓度侧向高浓度侧流动，直到两侧浓度平衡

推导链：
  1. 价格偏离均线 = 溶质浓度差（偏离越大，回归力越强）
  2. BIAS极端负值 = 低浓度侧（水分子净流入=资金净流入的潜力）
  3. RSI确认超卖 = 膜的通透性（超卖=膜打开，回归更容易发生）
  4. 放量确认 = 水分子开始流动（成交量=渗透速率）
  5. 入场：价格被极端拉离均线+超卖确认+放量确认回归
  6. 出场：价格回到均线附近（渗透平衡）

JSON输出：
{
  "science_concept": {
    "domain": "化学",
    "name": "渗透压·半透膜平衡",
    "mechanism": "溶质浓度差驱动分子从低浓度向高浓度流动直至平衡"
  },
  "derivation": "价格偏离均线=浓度差，BIAS极负=低浓度侧，RSI超卖=膜通透，放量=渗透开始，回归均线=平衡",
  "output_form": "new_strategy",
  "strategy_name": "渗透压乖离回归",
  "entry": "BIAS(60) < -0.08 AND RSI(14) < 25 AND volume_ratio(5) > 1.3 AND close < MA(60)",
  "exit": "BIAS(20) > -0.02 OR close > MA(20)",
  "direction": "long",
  "regime": "超跌反弹",
  "indicator_families": ["mean_reversion", "momentum", "volume"],
  "reason_for_failure": null
}

═══════════════════════════════════════════════════════════
示范2：热带气旋能量循环（气象学）→ 趋势衰竭反转（趋势衰竭型）
═══════════════════════════════════════════════════════════

科学概念：
  domain: 气象学
  name: 热带气旋能量循环·眼壁置换
  mechanism: 强台风眼壁收缩到极限后，外围形成新眼壁替换旧眼壁，期间台风强度先减弱再增强

推导链：
  1. 趋势加速后振幅突然收窄 = 眼壁收缩到极限（旧趋势能量耗尽）
  2. CCI从超买区拐头 = 强度减弱信号（旧眼壁瓦解）
  3. WR从极端值回升 = 外围新眼壁形成（新资金开始接盘）
  4. 入场：确认前一轮趋势能量耗尽，短期反转在即
  5. 出场：新趋势方向确认后离场

JSON输出：
{
  "science_concept": {
    "domain": "气象学",
    "name": "热带气旋·眼壁置换",
    "mechanism": "台风眼壁收缩到极限后外围新眼壁替换旧眼壁，强度先减后增"
  },
  "derivation": "趋势加速后振幅收窄=眼壁收缩，CCI超买拐头=强度减弱，WR从极值回升=新眼壁形成，短期反转",
  "output_form": "new_strategy",
  "strategy_name": "眼壁置换趋势衰竭",
  "entry": "pct_change(10) > 0.15 AND amplitude(5) < 0.04 AND CCI(14) < 150 AND WR(14) > -60 AND volume_ratio(5) > 1.0",
  "exit": "CCI(14) < 50 OR WR(14) > -20 OR close < MA(10)",
  "direction": "short",
  "regime": "衰竭反转",
  "indicator_families": ["momentum", "volatility", "trend"],
  "reason_for_failure": null
}

═══════════════════════════════════════════════════════════
示范3：动作电位（神经科学）→ 阈值放量突破（量价配合型）
═══════════════════════════════════════════════════════════

科学概念：
  domain: 神经科学
  name: 动作电位·全或无定律
  mechanism: 神经元膜电位必须超过阈值(-55mV)才能触发动作电位，低于阈值完全不触发，高于阈值全幅度触发

推导链：
  1. 价格在均线上方窄幅整理 = 静息电位（等待触发）
  2. ATR收缩到极低 + 成交量萎缩 = 膜电位接近阈值（能量积累）
  3. 成交量突然暴增 + 价格突破近期高点 = 达到阈值触发动作电位
  4. 关键：必须有"全或无"的跳跃——量必须突然放大（不能温和），价必须明确突破（不能犹豫）
  5. 入场：量价同时满足阈值条件时全仓进入
  6. 出场：量能回落到阈值以下（不应期）

JSON输出：
{
  "science_concept": {
    "domain": "神经科学",
    "name": "动作电位·全或无定律",
    "mechanism": "膜电位超过阈值全幅度触发，低于阈值完全不触发"
  },
  "derivation": "窄幅整理+低波=静息电位，量暴增+价突破=达阈值触发，量价必须同时跳变（全或无）",
  "output_form": "new_strategy",
  "strategy_name": "全或无量价突破",
  "entry": "ATR(20) / close < 0.02 AND volume_ratio(5) > 2.5 AND close > MA(60) AND close / MA(20) > 1.03 AND consecutive_up >= 2",
  "exit": "volume_ratio(5) < 1.0 OR close < MA(10) OR consecutive_down >= 3",
  "direction": "long",
  "regime": "突破",
  "indicator_families": ["volatility", "volume", "trend", "flow"],
  "reason_for_failure": null
}

═══════════════════════════════════════════════════════════
示范4：洛特卡-沃尔泰拉方程（生态学）→ 多空周期轮动（周期振荡型）
═══════════════════════════════════════════════════════════

科学概念：
  domain: 生态学
  name: 捕食者-猎物周期振荡
  mechanism: 猎物增多→捕食者增多→猎物减少→捕食者减少→猎物增多，形成相位差1/4周期的永续振荡

推导链：
  1. 超买超卖指标(RSI/WR)的周期性摆动 = 捕食者-猎物数量振荡
  2. RSI从低位上穿30 + WR从高位回落 = 猎物开始增多（多头萌芽）
  3. RSI在50-70区间 + WR在-50~-30 = 捕食者主导（趋势加速）
  4. RSI上穿70 + WR下穿-20 = 捕食者过多猎物耗竭（超买反转预警）
  5. 入场：确认猎物增多阶段（RSI刚从低位回升+WR确认）
  6. 出场：捕食者过度捕猎（RSI超买+WR极端+量能背离）

JSON输出：
{
  "science_concept": {
    "domain": "生态学",
    "name": "捕食者-猎物周期振荡",
    "mechanism": "猎物与捕食者数量形成相位差1/4周期的永续振荡"
  },
  "derivation": "RSI/WR摆动=种群数量振荡，RSI低升+WR高落=猎物增多，RSI超买+WR极低=捕食者过剩",
  "output_form": "new_strategy",
  "strategy_name": "捕食者周期轮动",
  "entry": "RSI(14) > 30 AND RSI(14) < 45 AND WR(14) < -50 AND volume_ratio(5) > 1.0 AND close > MA(60)",
  "exit": "RSI(14) > 75 OR (WR(14) > -20 AND volume_ratio(5) < 0.8) OR CCI(14) > 200",
  "direction": "long",
  "regime": "周期振荡",
  "indicator_families": ["momentum", "volume"],
  "reason_for_failure": null
}
"""

LLM_SYSTEM_PROMPT = f"""你是一个跨学科量化策略研究员。你的核心能力是将自然科学概念映射到股票市场，推导出可验证的交易策略。

工作流程：
1. 随机选择一个自然科学概念（物理、化学、生物、天文、地质、医学...任意领域）
2. 理解这个概念的核心机制（公式、定律、结构、过程）
3. 思考这个机制在股票市场上对应什么可观测现象
4. 设计一个能用日线数据计算和验证的交易策略
5. 写出可计算的 IF-THEN 条件表达式

关键原则：
- 疯狂的联想是允许的，但推导链必须清晰
- 概念 → 市场映射 → 入场条件 → 出场条件，每一步都要解释"为什么"
- 如果某个概念实在无法映射，诚实返回 {{"strategy_name": null, "reason_for_failure": "..."}}
- 不要硬凑。质量 > 数量
- 每个策略的输出必须是可计算的条件表达式
- **重要**：书写表达式时不要使用 .shift() 语法，应使用带参数的函数形式。例如表达"MA5上穿MA20"应写为 MA(5) > MA(20)

【表达式差异化要求 — 最重要！违反将导致策略无效】
你的表达式必须体现科学概念的独特特征。以下模板已被过度使用，禁止直接套用：
  ❌ 禁止: amplitude(20) < X AND volume_ratio(5) > Y AND close > MA(60)  — 这是"振幅突破模板"，已被使用81次
  ❌ 禁止: bollinger_width(20) < X AND close > MA(20) AND volume_ratio(5) > Y  — "布林收缩模板"
  ❌ 禁止: MA(5) > MA(20) AND MA(20) > MA(60) AND volume_ratio(5) > X  — "多周期共振模板"
  ❌ 禁止: RSI(14) < 35 AND volume_ratio(5) > X AND close > MA(60)  — "超跌反弹模板"
如果你要使用类似模式，必须证明该科学概念严格对应此模式，且至少使用2个以上模板中未出现的指标（如CCI、WR、ATR比值、consecutive、形态函数等）来差异化。

【参数选择要求】
- 每个参数的数值必须有科学依据（例如：渗透压里BIAS=-0.08对应"浓度差阈值"，而不能随便填-0.05）
- 在derivation字段中解释关键参数的科学含义
- 尽量使用指标之间的比值、差值、交叉关系，而不是简单的阈值比较
- 多指标组合时，每个指标应该从不同角度验证同一个科学假说（如同一个物理过程的不同可观测量）

【策略类型（regime）— 必填，输出到JSON的regime字段】
从以下6种分类中选择最适合的一个：
- "超跌反弹": 价格过度下跌后的反弹行情（BIAS极端+RSI超卖+形态确认）
- "趋势跟随": 顺势交易，跟踪已形成的趋势方向（MA多头排列+斜率确认）
- "突破": 等待关键价位或波动率收缩后被突破（布林收缩+放量突破）
- "衰竭反转": 在趋势末期捕捉反转信号（趋势减速+波动率突变+指标背离）
- "量价背离": 利用量价关系的不一致进行交易（价涨量缩/价跌量增）
- "周期振荡": 利用指标的周期性摆动进行波段操作（RSI/WR/CCI周期相位）

【indicator_families — 必填，输出到JSON的indicator_families字段】
列出entry_expr中使用的主要指标家族（1-4个），从以下选择：
- mean_reversion: BIAS, MA, ma_spread, BOLL, ENE
- momentum: RSI, CCI, WR, MACD, KDJ, ADX, TRIX, ULTOSC
- volume: volume_ratio, OBV, MFI, AD, VR, PVT, WVAD
- volatility: ATR, NATR, amplitude, std, BOLL宽度, CHOP
- trend: MA_slope, SUPERTREND, DMI, AROON, SAR, LINEARREG_SLOPE
- pattern: is_hammer, CDLHAMMER, CDLENGULFING等K线形态
- flow: consecutive_up, consecutive_down, up_days, down_days, gap_up, gap_down

【鼓励的策略类型（优先选择未被过度使用的模式）】
- RSI/WR/CCI之间的背离与共振（如RSI上升但CCI下降 → 背离）
- ATR多周期比值（如ATR(5)/ATR(20) → 短中长期波动率变化）
- 形态+指标的组合确认（如is_hammer() + CCI极值 + WR确认）
- 连涨/连跌+量能背离（如consecutive_up >= 3 AND volume_trend(5) < 0）
- BIAS多周期共振（如BIAS(20) < -0.03 AND BIAS(60) < -0.05 → 多周期超跌共振）
- 做空信号（is_shooting_star + RSI超买 + CCI确认）

可用数据字段：close, open, high, low, volume。
可用指标函数（签名中的参数名表示位置参数顺序；无参数函数不得传参）：
{indicator_catalog_text()}

条件表达式语法：
- 比较运算符：>, <, >=, <=, ==, !=
- 逻辑运算符：AND, OR, NOT
- 数学运算：+, -, *, /, abs()
- 示例：BIAS(60) < -0.08 AND RSI(14) < 25 AND volume_ratio(5) > 1.3 AND close < MA(60)

下面是4个不同类型的完整示范（均值回归型、趋势衰竭型、量价突破型、周期振荡型）：

{FEW_SHOT_EXAMPLES}

请随机选择一个自然科学概念，按上述格式输出JSON。

【关键规则 - 必须严格遵守】
1. 你的整个回复必须是一个合法的JSON对象，以{{开头，以}}结尾
2. 不要输出任何思考过程、解释、前缀或后缀文字
3. 不要在JSON前后添加任何文字
4. 不要用```json```包裹
5. 如果无法映射，返回{{"strategy_name": null, "reason_for_failure": "原因"}}

错误示例（禁止）："好的，我来分析一下...{{...}}"
正确示例："{{"science_concept": {{...}}, "strategy_name": "...", ...}}"""


def _build_llm_user_prompt(domain: str) -> str:
    """Build a domain-specific prompt with anti-template guidance."""
    # Domain-specific hints to encourage diverse strategy types
    domain_hints = {
        "物理学": "优先考虑动量、惯性、共振、波的干涉/衍射 → 尝试MA_slope比值、多周期CCI共振、ATR波动率传播",
        "化学": "优先考虑平衡、催化、反应速率、渗透 → 尝试BIAS回归、量能催化（volume_ratio突变）、多周期BIAS共振",
        "生物学": "优先考虑周期、共生、竞争、适应 → 尝试RSI/WR周期相位差、多指标共生确认、连涨连跌适应",
        "天文学": "优先考虑轨道、引力、周期、坍缩 → 尝试ma_spread轨道宽度、BIAS引力回归、振幅坍缩",
        "地质学": "优先考虑板块运动、地震、侵蚀、沉积 → 尝试趋势碰撞（MA交叉）、振幅突变（地震）、量能侵蚀",
        "医学/生理学": "优先考虑稳态、反射、免疫、节律 → 尝试BIAS稳态回归、形态反射（is_hammer+CCI确认）、多周期节律",
        "数学": "优先考虑极限、收敛、分形、概率 → 尝试波动率收敛（ATR/std比值）、分形多周期、bollinger_width概率回归",
        "生态学": "优先考虑演替、生态位、食物网 → 尝试阶段转换（amplitude+CCI配合）、生态位分化（多指标各管一维）",
        "流体力学": "优先考虑层流/湍流、压力差、涡旋 → 尝试ATR比值（雷诺数）、量能压力、振幅涡旋",
        "热力学": "优先考虑熵增、热传导、相变 → 尝试趋势衰减（MA_slope递减）、波动率相变（std突变）、量能热传导",
        "电磁学": "优先考虑感应、滞后、共振 → 尝试量价感应（volume_ratio+price背离）、均线滞后回归",
        "量子力学": "优先考虑隧穿、叠加、坍缩、不确定性 → 尝试布林带突破（隧穿）、多指标叠加确认、振幅坍缩",
        "遗传学": "优先考虑突变、选择、遗传 → 尝试突变检测（gap_up+volume极端）、自然选择（MA_slope多周期筛选）",
        "神经科学": "优先考虑阈值、可塑性、信号传导 → 尝试阈值突破（ATR+CCI配合）、趋势记忆强化（MA_slope持续）",
        "气象学": "优先考虑锋面、气旋、对流 → 尝试冷锋（pct_change减速+CCI转弱）、对流（量价振幅共振）",
    }
    hint = domain_hints.get(domain, "优先使用多指标组合，避免单一模板")

    return (
        f"从[{domain}]领域选一个概念，推导交易策略。\n\n"
        f"领域提示: {hint}\n\n"
        "【重要】不要使用以下模板：amplitude+volume_ratio+close>MA、纯多周期共振、纯超跌反弹RSI+volume_ratio。"
        "你的表达式必须体现科学概念的独特特征。至少使用2个不同类型的指标（如CCI+WR、ATR比值+形态、BIAS多周期+量能背离）。\n\n"
        "严格输出JSON（不要思考过程，不要解释，不要```json```包裹，整个回复必须以{开头以}结尾）。\n\n"
        '如果无法映射：{"strategy_name": null, "reason_for_failure": "原因"}'
    )


# ═══════════════════════════════════════════════════════════════
# Legacy combo constants (used by fallback mode when LLM unavailable)
# ═══════════════════════════════════════════════════════════════

_COMBO_AND = "AND"
_COMBO_OR = "OR"
_COMBO_SEQ = "时序"
_COMBO_FILTER = "过滤"
_LEGACY_COMBO_TYPES = [_COMBO_AND, _COMBO_OR, _COMBO_SEQ, _COMBO_FILTER]

_LEGACY_COMBO_TEMPLATES = {
    _COMBO_AND: {
        "name_fmt": "{a} ∧ {b} 共振",
        "condition_fmt": "{cond_a} AND {cond_b}",
        "logic_fmt": "同时满足{a}和{b}的入场条件时买入",
    },
    _COMBO_OR: {
        "name_fmt": "{a} ∨ {b} 并行",
        "condition_fmt": "{cond_a} OR {cond_b}",
        "logic_fmt": "{a}或{b}任一满足入场条件时买入，扩大信号覆盖",
    },
    _COMBO_SEQ: {
        "name_fmt": "{a} → {b} 接力",
        "condition_fmt": "{cond_a} 信号发出后 3 天内 {cond_b} 也触发",
        "logic_fmt": "{a}出信号后3天内{b}也出信号则加仓，捕捉趋势延续",
    },
    _COMBO_FILTER: {
        "name_fmt": "{a} + {b}过滤",
        "condition_fmt": "{cond_a} 选股后，用 {cond_b} 做二次筛选",
        "logic_fmt": "{a}初选股票池，{b}条件过滤排除假信号",
    },
}


class L1ContrastiveReasoner:
    """Generates hypotheses by mapping natural-science concepts to trading strategies.

    Two modes:
    - Natural-law mode (SPARK_LLM=1): LLM freely associates science concepts
      with market phenomena and writes executable entry/exit expressions.
    - Fallback mode (no LLM): original strategy-pairing logic preserved.
    """

    def __init__(self, kb):
        self.kb = kb
        self.compiler = ExpressionCompiler()
        self._tried_concepts: set = set()      # (domain, concept_name)
        self._tried_expressions: set = set()   # expression hashes
        self._used_combo_names: set = set()
        self._consecutive_failures: int = 0    # for LLM cooldown
        self._unified_expressions: set[str] = set()

        # Load existing combo names from KB to avoid regenerating duplicates
        for h in kb.list_hypotheses():
            if h.generation_layer == 1 and h.name:
                self._used_combo_names.add(h.name)
            if h.generation_layer == 1 and h.condition_expr:
                self._tried_expressions.add(hashlib.md5(h.condition_expr.encode()).hexdigest()[:12])
                self._unified_expressions.add(h.condition_expr)

        # Persisted verified L1 expressions remain available to L2 parameter
        # exploration after an engine restart.
        for h in kb.list_hypotheses(status=HypothesisStatus.VERIFIED):
            condition = h.condition_expr or ""
            if h.generation_layer == 1 and condition and self.compiler.validate(condition)[0]:
                self._register_dynamic_strategy(
                    h.name,
                    self.compiler.make_scan_function(condition, None),
                )

    # ── public API ──────────────────────────────────────────────

    def generate(self, max_hypotheses: int = 4) -> list[Hypothesis]:
        """Generate L1 hypotheses. Reads from natural_law_ideas table first,
        falls back to legacy strategy pairing if the idea pool is empty.
        """
        if self._consecutive_failures < 5:
            result = self._generate_natural_law(max_hypotheses)
            if result:
                return result
        # Natural-law pool is optional. The registry-driven generator keeps L1
        # alive without an external LLM and uses the exact same expression DSL
        # as L3/L4 and the unified verifier.
        try:
            from spark_core.unified_engine import generate_unified_hypotheses, seed_generated_expressions
            from backtest_framework import scan_all_csvs
            seed_generated_expressions(self._unified_expressions)
            unified = generate_unified_hypotheses(
                max_hypotheses=max_hypotheses,
                generation_layer=1,
                sample_csv_files=scan_all_csvs(),
            )
            if unified:
                print(f"[L1] 统一指标注册表生成 {len(unified)} 个表达式策略")
                for h in unified:
                    self._register_dynamic_strategy(
                        h.name,
                        self.compiler.make_scan_function(h.condition_expr, None),
                    )
                return unified
        except Exception as e:
            print(f"[L1] 统一指标生成失败: {e}")
        return self._generate_legacy(max_hypotheses)

    # ── Natural-law mode ────────────────────────────────────────

    def _generate_natural_law(self, max_hypotheses: int) -> list[Hypothesis]:
        """Read pre-generated ideas from natural_law_ideas table.

        Ideas are batch-generated by batch_generate_ideas.py and stored in SQLite.
        This method reads unused ideas, compiles expressions, and creates Hypotheses.
        Falls back to legacy mode if the idea pool is empty.
        """
        hypotheses = []

        try:
            from spark_ledger import ExperimentLedger
            db_path = os.environ.get("SPARK_DB_PATH", "")
            ideas = (
                ExperimentLedger(db_path).unused_natural_law_ideas(limit=max_hypotheses)
                if db_path else []
            )
        except Exception:
            ideas = []

        if not ideas:
            print("[L1] 自然规律想法池已空，回退到传统配对模式。运行 batch_generate_ideas.py 补充。")
            self._consecutive_failures = 5  # Trigger legacy fallback
            return hypotheses

        for idea in ideas:
            entry_expr = idea.get("entry_expr", "")
            exit_expr = idea.get("exit_expr", "")
            strategy_name = idea.get("strategy_name", "")
            idea_id = idea.get("id")

            if not entry_expr or not strategy_name:
                if idea_id:
                    db.mark_natural_law_idea_used(idea_id)
                continue

            # Validate expression
            ok, err = self.compiler.validate(entry_expr)
            if not ok:
                print(f"[L1] 想法 '{strategy_name}' 表达式校验失败: {err}")
                if idea_id:
                    db.mark_natural_law_idea_used(idea_id)
                continue

            try:
                from spark_core.unified_engine import (
                    MAX_INITIAL_CONDITION_COUNT,
                    parse_boolean,
                )

                def _leaf_count(node):
                    if node.op == "leaf":
                        return 1
                    return sum(_leaf_count(child) for child in node.children or [])

                if _leaf_count(parse_boolean(entry_expr)) > MAX_INITIAL_CONDITION_COUNT:
                    if idea_id:
                        db.mark_natural_law_idea_used(idea_id)
                    continue
            except Exception:
                pass

            if exit_expr:
                ok, _ = self.compiler.validate(exit_expr)
                if not ok:
                    exit_expr = ""

            # Dedup
            expr_hash = hashlib.md5(entry_expr.encode()).hexdigest()[:12]
            if expr_hash in self._tried_expressions:
                if idea_id:
                    db.mark_natural_law_idea_used(idea_id)
                continue
            self._tried_expressions.add(expr_hash)

            if strategy_name in self._used_combo_names:
                if idea_id:
                    db.mark_natural_law_idea_used(idea_id)
                continue
            self._used_combo_names.add(strategy_name)

            # Parameter optimization — LLM proposes structure, we search best params
            optimizer = self._get_param_optimizer()
            if optimizer is not None:
                regime = idea.get("regime", "")
                opt_best_expr, opt_stats = optimizer.optimize(
                    entry_expr, regime=regime, direction=idea.get("direction", "long")
                )
                if opt_best_expr and opt_best_expr != entry_expr:
                    old_pf = opt_stats.get("original_pf", 0)
                    new_pf = opt_stats.get("best_pf", 0)
                    if new_pf > old_pf + 0.01:
                        print(f"[L1] 🔧 {strategy_name}: PF {old_pf:.2f} → {new_pf:.2f} "
                              f"({opt_stats.get('combinations_tested', 0)} tests, "
                              f"{opt_stats.get('search_time_s', 0):.0f}s)")
                        if self._debug:
                            print(f"     原始: {entry_expr[:100]}")
                            print(f"     优化: {opt_best_expr[:100]}")
                        entry_expr = opt_best_expr

            # Compile scan function
            scan_func = self.compiler.make_scan_function(entry_expr, exit_expr if exit_expr else None)
            self._register_dynamic_strategy(strategy_name, scan_func)

            # Build Hypothesis
            direction = Direction.LONG if idea.get("direction") == "long" else Direction.SHORT
            description = (
                f"[{idea.get('science_domain', '')}·{idea.get('concept_name', '')}] "
                f"{idea.get('mechanism', '')}。推导：{idea.get('derivation', '')}。"
                f"入场条件：{entry_expr[:120]}"
            )[:300]

            h = Hypothesis(
                direction=direction,
                name=strategy_name,
                description=description,
                condition_expr=entry_expr,
                signal_logic_desc=idea.get("derivation", ""),
                source_chain=[
                    idea.get("science_domain", ""),
                    idea.get("concept_name", ""),
                ],
                generation_layer=1,
                combo_type=idea.get("output_form", "new_strategy"),
            )
            hypotheses.append(h)

            # Mark as used
            if idea_id:
                try:
                    db.mark_natural_law_idea_used(idea_id)
                except Exception:
                    pass

        if hypotheses:
            remaining = db.count_unused_ideas() if hasattr(db, 'count_unused_ideas') else '?'
            print(f"[L1] 从想法池消费 {len(hypotheses)} 个策略，剩余 {remaining} 个")
        else:
            print("[L1] 想法池中没有可用的有效策略")

        return hypotheses

    def _call_llm_for_natural_law(self, domain: str) -> str:
        """Call LLM with the natural-law few-shot prompt."""
        return llm_reason(
            system_prompt=LLM_SYSTEM_PROMPT,
            user_prompt=_build_llm_user_prompt(domain),
            timeout=90,
        )

    def _parse_llm_output(self, text: str) -> dict | None:
        """Parse JSON from LLM output. Handles markdown code blocks and
        verbose DeepSeek-style responses that contain reasoning before JSON."""
        if not text:
            return None
        text = text.strip()

        # Remove markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3].strip()

        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 1: Find the first '{' and last '}', try to parse that slice
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            candidate = text[first_brace:last_brace + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Strategy 2: Try to find JSON by looking for "science_concept" key
        match = re.search(
            r'\{(?:[^{}]|\{[^{}]*\})*"science_concept"(?:[^{}]|\{[^{}]*\})*\}',
            text, re.DOTALL
        )
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Try regex for "strategy_name" key
        match = re.search(r'\{[^}]*"strategy_name"[^}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _record_failed_analogy(self, parsed: dict):
        """Log when LLM says a concept cannot be mapped."""
        concept = parsed.get("science_concept", {})
        reason = parsed.get("reason_for_failure", "")
        print(f"[L1] LLM 无法映射: {concept.get('name', '?')} — {reason[:120]}")

    def _record_compile_failure(self, parsed: dict, error: str):
        """Log expression compile failures to failure_museum if available."""
        try:
            from spark_core.failure_museum import FailureMuseum
            fm = FailureMuseum()
            fm.add_compile_failure(
                science_concept=parsed.get("science_concept", {}),
                strategy_name=parsed.get("strategy_name", "?"),
                entry=parsed.get("entry", ""),
                error=error,
            )
        except Exception:
            pass

    def _build_description(self, parsed: dict) -> str:
        """Build a human-readable hypothesis description from parsed JSON."""
        concept = parsed["science_concept"]
        domain = concept.get("domain", "")
        name = concept.get("name", "")
        mechanism = concept.get("mechanism", "")
        derivation = parsed.get("derivation", "")
        return (
            f"[{domain}·{name}] {mechanism}。推导：{derivation}。"
            f"入场条件：{parsed['entry'][:120]}"
        )[:300]

    def _get_param_optimizer(self):
        """Lazy-init ParamOptimizer. Returns None if disabled or unavailable."""
        if getattr(self, '_opt_disabled', False):
            return None
        if not hasattr(self, '_param_optimizer'):
            env_opt = os.environ.get("SPARK_PARAM_OPT", "1")
            if env_opt in ("0", "false", "no", "off"):
                self._opt_disabled = True
                return None
            try:
                from spark_core.spark_l1_param_optimizer import ParamOptimizer
                self._param_optimizer = ParamOptimizer(sample_n=200)
                print("[L1] ParamOptimizer 已初始化 (sample_n=200)")
            except Exception as e:
                print(f"[L1] ParamOptimizer 初始化失败: {e}, 跳过优化")
                self._opt_disabled = True
                return None
        return self._param_optimizer

    def _register_dynamic_strategy(self, name: str, scan_func):
        """Register a dynamically-created strategy for L2 and verification.
        Uses an initial PF of 1.0 (unknown); real PF is backfilled after backtest.
        """
        DYNAMIC_STRATEGIES[name] = (scan_func, 1.0, 10)
        # Also register in KB so L2 knows about it
        if hasattr(self.kb, 'register_dynamic_strategy_node'):
            self.kb.register_dynamic_strategy_node(name, {
                "grade": "?",
                "pf": 1.0,
                "win_rate": 50.0,
                "group": "自然规律",
                "hold": 10,
                "tp": 10,
                "sl": -8,
                "dynamic": True,
            })

    # ── Fallback: legacy strategy pairing ───────────────────────

    def _generate_legacy(self, max_hypotheses: int = 3) -> list[Hypothesis]:
        """Original strategy-pairing logic. Used when LLM is unavailable."""
        analogies = list(self.kb.analogies.analogies.values())
        strategies = [
            n for n in self.kb.graph.nodes.values()
            if n.node_type == "strategy"
        ]
        if len(strategies) < 2:
            return []

        hypotheses = []
        for _ in range(min(max_hypotheses, len(analogies))):
            analogy = random.choice(analogies)
            mapping = self._map_analogy_to_strategies(analogy, strategies)
            for pair in mapping[:1]:
                strat_a = pair.get("strategy_a", "")
                strat_b = pair.get("strategy_b", "")
                combo_type = pair.get("combo_type", _COMBO_OR)

                if combo_type in (_COMBO_AND, _COMBO_SEQ, _COMBO_FILTER):
                    if not self._check_combo_viable(strat_a, strat_b, combo_type):
                        combo_type = _COMBO_OR

                tmpl = _LEGACY_COMBO_TEMPLATES.get(combo_type, _LEGACY_COMBO_TEMPLATES[_COMBO_AND])
                name = tmpl["name_fmt"].format(a=strat_a, b=strat_b)
                if name in self._used_combo_names:
                    continue
                self._used_combo_names.add(name)

                cond_a = f"{strat_a}入场信号"
                cond_b = f"{strat_b}入场信号"
                condition = tmpl["condition_fmt"].format(cond_a=cond_a, cond_b=cond_b)
                logic = tmpl["logic_fmt"].format(a=strat_a, b=strat_b)

                h = Hypothesis(
                    direction=Direction.LONG,
                    name=name,
                    description=(
                        f"[{analogy.source_domain}·{analogy.source_pattern}] "
                        f"{analogy.finance_mapping}。{logic}"
                    ),
                    condition_expr=condition,
                    signal_logic_desc=logic,
                    source_chain=[analogy.id, strat_a, strat_b],
                    generation_layer=1,
                    combo_type=combo_type,
                )
                hypotheses.append(h)
        return hypotheses

    # ── Legacy methods (kept for fallback) ──────────────────────

    def _map_analogy_to_strategies(self, analogy, strategies) -> list[dict]:
        strategy_names = [s.name for s in strategies]
        strategy_list = "\n".join(f"- {n}" for n in strategy_names)

        llm_result = llm_reason(
            system_prompt=(
                "你是一个量化策略研究员。给定一个跨学科类比和可用的策略列表，"
                "将类比映射到具体的策略组合。返回 JSON 数组，每个元素包含："
                "strategy_a(策略名), strategy_b(策略名), combo_type(优先OR，只在策略信号高度相关时用AND/时序/过滤), rationale(一句话理由)。"
                "最多返回 3 个组合。只返回 JSON，不要其他文字。"
            ),
            user_prompt=(
                f"跨学科类比：「{analogy.source_domain}·{analogy.source_pattern}」— {analogy.finance_mapping}\n"
                f"示例：{analogy.examples[0] if analogy.examples else ''}\n\n"
                f"可用策略列表：\n{strategy_list}\n\n"
                f"请返回 2-3 个策略组合 JSON 数组。"
            ),
        )

        if llm_result:
            try:
                llm_result = llm_result.strip()
                if llm_result.startswith("```"):
                    llm_result = llm_result.split("\n", 1)[1].rsplit("\n```", 1)[0]
                parsed = json.loads(llm_result)
                if isinstance(parsed, list) and len(parsed) > 0:
                    valid = []
                    for p in parsed:
                        a = p.get("strategy_a", "")
                        b = p.get("strategy_b", "")
                        if a in strategy_names and b in strategy_names:
                            valid.append(p)
                    if valid:
                        return valid
            except (json.JSONDecodeError, ValueError):
                pass
        return self._fallback_pairing(strategies)

    def _fallback_pairing(self, strategies) -> list[dict]:
        pairs = []
        by_group = {}
        for s in strategies:
            g = s.metadata.get("group", "其他")
            by_group.setdefault(g, []).append(s)
        groups = list(by_group.keys())
        random.shuffle(groups)
        for i in range(min(3, len(groups))):
            if len(by_group[groups[i]]) >= 2:
                sa, sb = random.sample(by_group[groups[i]], 2)
                combo = random.choice(["OR", "OR", "OR", "OR", "OR", "OR", "OR", "AND", "时序", "过滤"])
                pairs.append({
                    "strategy_a": sa.name,
                    "strategy_b": sb.name,
                    "combo_type": combo,
                    "rationale": f"同组组合：{groups[i]}",
                })
            else:
                g_b = groups[(i + 1) % len(groups)] if len(groups) > 1 else groups[0]
                sa = random.choice(by_group[groups[i]])
                sb = random.choice(by_group[g_b])
                if sa.name == sb.name:
                    continue
                pairs.append({
                    "strategy_a": sa.name,
                    "strategy_b": sb.name,
                    "combo_type": "OR",
                    "rationale": f"跨组组合：{groups[i]} × {g_b}",
                })
        return pairs

    def _check_combo_viable(self, strat_a: str, strat_b: str, combo_type: str) -> bool:
        if combo_type == "OR":
            return True
        try:
            from spark_core.verification_pipeline import _get_strategy_scan_map, _run_single_scan
            scan_map = _get_strategy_scan_map()
            if strat_a not in scan_map or strat_b not in scan_map:
                return True
            from backtest_framework import scan_all_csvs
            csv_files = scan_all_csvs()
            if not csv_files:
                return True
            sample_n = min(30, len(csv_files))
            sample = random.sample(csv_files, sample_n)
            sigs_a = _run_single_scan(scan_map[strat_a], sample)
            sigs_b = _run_single_scan(scan_map[strat_b], sample)
            if not sigs_a or not sigs_b:
                return False
            dates_a = {(s['code'], str(s.get('signal_date', ''))) for s in sigs_a}
            dates_b = {(s['code'], str(s.get('signal_date', ''))) for s in sigs_b}
            overlap = dates_a & dates_b
            return len(overlap) >= 3
        except Exception:
            return True
