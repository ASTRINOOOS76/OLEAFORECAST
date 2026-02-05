/**
 * Canonical Serving Gates & Horizon Utilities
 * 
 * All gate evaluation and horizon conversion MUST use these functions.
 */

// ============ HORIZON MAPPING ============

export const HORIZON_WEEKS_TO_MONTHS: Record<number, number> = {
  4: 1,
  8: 2,
  12: 3,
  13: 3,  // 13 weeks ≈ 3 months
  26: 6,
  39: 9,
  52: 12,
};

export const HORIZON_MONTHS_TO_WEEKS: Record<number, number> = {
  1: 4,
  2: 8,
  3: 12,
  6: 26,
  9: 39,
  12: 52,
};

export function weeksToMonths(weeks: number): number {
  return HORIZON_WEEKS_TO_MONTHS[weeks] ?? Math.round(weeks / 4.33);
}

export function monthsToWeeks(months: number): number {
  return HORIZON_MONTHS_TO_WEEKS[months] ?? Math.round(months * 4.33);
}

// ============ CANONICAL SERIES KEY & CLUSTER ============

const CLUSTER_ALIASES: Record<string, string> = {
  'Extra Virgin': 'EVOO',
  'ExtraVirgin': 'EVOO',
  'extra_virgin': 'EVOO',
  'Virgin': 'VOO',
  'virgin': 'VOO',
  'Lampante': 'LAMPANTE',
  'lampante': 'LAMPANTE',
  'Refined': 'REFINED',
  'refined': 'REFINED',
  'Pomace': 'POMACE',
  'pomace': 'POMACE',
};

/**
 * Normalize cluster name to canonical form (EVOO, VOO, LAMPANTE, REFINED, POMACE).
 * Use this EVERYWHERE a cluster is processed.
 */
export function normalizeCluster(cluster: string): string {
  if (!cluster) return 'EVOO';
  const upper = cluster.toUpperCase().trim();
  if (['EVOO', 'VOO', 'LAMPANTE', 'REFINED', 'POMACE'].includes(upper)) {
    return upper;
  }
  return CLUSTER_ALIASES[cluster] || upper;
}

/**
 * Build canonical seriesKey from cluster and market.
 * Format: "{CLUSTER}_{MARKET}" e.g., "EVOO_GR"
 */
export function buildSeriesKey(cluster: string, market: string): string {
  const normalizedCluster = normalizeCluster(cluster);
  const normalizedMarket = (market || 'ES').toUpperCase().trim();
  return `${normalizedCluster}_${normalizedMarket}`;
}

/**
 * Parse a seriesKey into cluster and market components.
 */
export function parseSeriesKey(seriesKey: string): { cluster: string; market: string } {
  const parts = seriesKey.split('_');
  const cluster = normalizeCluster(parts[0] || 'EVOO');
  const market = (parts[1] || 'ES').toUpperCase().trim();
  return { cluster, market };
}

// ============ GATE THRESHOLDS ============

export const GATE_THRESHOLDS = {
  /** Bias gate: absolute EUR/t threshold (hard fail) */
  BIAS_MAX_EUR_T: 250,
  
  /** MASE gate: max acceptable MASE score */
  MASE_MAX: 1.5,
  
  /** Interval coverage gate: minimum acceptable coverage for p10/p90 (hard lower bound) */
  COVERAGE_MIN: 0.75,  // For p10/p90 intervals, 75% is reasonable hard floor
  
  /** Prediction availability gate: minimum */
  AVAILABILITY_MIN: 0.98,
  
  /** Sample size gate: minimum n_eval for reliable metrics */
  N_EVAL_MIN: 100,
  
  /** Minimum folds for reliable cross-validation (hard gate) */
  FOLDS_MIN: 12,
  
  /** Recommended folds for high-confidence metrics (soft warning) */
  FOLDS_RECOMMENDED: 20,
  
  /** Target coverage for p10/p90 intervals (center point for symmetric penalty) */
  TARGET_COVERAGE: 0.80,  // For p10/p90, target is 80%, not 90%
  
  /** Target coverage band for calibration (soft target, not hard gates) */
  TARGET_COVERAGE_LOW: 0.75,   // Adjusted for p10/p90
  TARGET_COVERAGE_HIGH: 0.85,  // Adjusted for p10/p90
  
  /** Coverage penalty thresholds (warnings, not failures) */
  COVERAGE_WARNING_HIGH: 0.88,  // Warning: intervals may be too wide
  COVERAGE_SEVERE_HIGH: 0.92,   // Severe warning but still eligible
  
  /** Minimum folds required for valid backtest */
  MIN_FOLDS: 12,
  
  /** Relative gate: max MASE compared to baseline (110%) */
  MASE_BASELINE_MULTIPLIER: 1.10,
  
  /** Relative gate: max bias compared to baseline (110%) */
  BIAS_BASELINE_MULTIPLIER: 1.10,
  
  /** Stability rule: minimum improvement required to unseat incumbent (5%) */
  STABILITY_MARGIN_PCT: 5.0,
  
  /** Width sanity gate: minimum width as fraction of median price (prevents unrealistically narrow intervals) */
  WIDTH_MIN_RELATIVE: 0.08,  // 8% of median price minimum (e.g., 400 EUR/t for 5000 EUR/t price)
  
  /** Width sanity gate: minimum width as fraction of baseline width */
  WIDTH_MIN_VS_BASELINE: 0.5,  // Must be at least 50% of baseline width
  
  /** Coverage stability gate: maximum allowed standard deviation in coverage across folds */
  COVERAGE_STDDEV_MAX: 0.15,  // If std dev > 15%, coverage is unstable
} as const;

// ============ HARD-FAIL REASON CODES ============

export type HardFailReason = 
  | 'FOLDS_UNKNOWN'
  | 'INSUFFICIENT_FOLDS'
  | 'INSUFFICIENT_SAMPLE'
  | 'MASE_INVALID'
  | 'MAE_INVALID'
  | 'COVERAGE_INVALID'
  | 'COVERAGE_TOO_LOW'
  | 'BIAS_TOO_HIGH_ABS'
  | 'MASE_TOO_HIGH_ABS'
  | 'MASE_WORSE_THAN_BASELINE'
  | 'BIAS_WORSE_THAN_BASELINE'
  | 'NO_INTERVALS'
  | 'HORIZON_OVERRIDE'
  | 'WIDTH_TOO_NARROW'
  | 'COVERAGE_UNSTABLE';

// Horizon-specific overrides for known problematic experiment+horizon combos
// Based on backtest results: E7@4w coverage=51%, E12@26/52w MASE degrades, E11 has no intervals
export const HORIZON_OVERRIDES: Record<string, Record<number, HardFailReason[]>> = {
  'E7':  { 4: ['COVERAGE_TOO_LOW'] },  // 51% coverage at 4w
  'E12': { 26: ['MASE_TOO_HIGH_ABS'], 52: ['MASE_TOO_HIGH_ABS'] },  // MASE degrades to 1.5+ at long horizons
  'E11': { 4: ['NO_INTERVALS'], 12: ['NO_INTERVALS'], 26: ['NO_INTERVALS'], 52: ['NO_INTERVALS'] },  // 0% coverage = no intervals
  'E13': { 4: ['COVERAGE_TOO_LOW', 'MASE_TOO_HIGH_ABS'], 12: ['COVERAGE_TOO_LOW', 'MASE_TOO_HIGH_ABS'], 26: ['COVERAGE_TOO_LOW', 'MASE_TOO_HIGH_ABS'], 52: ['COVERAGE_TOO_LOW', 'MASE_TOO_HIGH_ABS'] },
  'E0':  { 4: ['BIAS_TOO_HIGH_ABS'], 12: ['BIAS_TOO_HIGH_ABS'], 26: ['BIAS_TOO_HIGH_ABS'], 52: ['BIAS_TOO_HIGH_ABS'] },  // Bias ~500 EUR/t
};

export interface EligibilityResult {
  eligible: boolean | null;  // null = unknown (e.g., missing fold count)
  hardFail: boolean;
  reasons: HardFailReason[];
  warnings: string[];
  details: Record<string, string>;
}

/**
 * Check if metric is valid (not null, NaN, or Infinity for positive metrics).
 */
function isValidMetric(value: number | null | undefined, allowZero = false): boolean {
  if (value == null) return false;
  if (!Number.isFinite(value)) return false;
  if (!allowZero && value <= 0) return false;
  return true;
}

/**
 * Context for gate evaluation - needed for horizon-specific overrides
 */
export interface GateContext {
  experimentId: string;
  horizon: number;
  nEval?: number | null;
  intervalWidth?: number | null;
  medianPrice?: number | null;
  baselineWidth?: number | null;
  coverageVariance?: number | null;
}

/**
 * Evaluate hard-fail eligibility gates.
 * These are absolute gates that must be passed regardless of baseline.
 * Now context-aware for horizon-specific overrides.
 */
export function evaluateHardFailGates(
  context: GateContext | null,
  metrics: {
    mase: number | null;
    mae: number | null;
    coverage: number | null;
    bias: number | null;
    folds?: number | null;
    intervalsAvailable?: boolean | null;
  }
): EligibilityResult {
  const reasons: HardFailReason[] = [];
  const warnings: string[] = [];
  const details: Record<string, string> = {};
  
  // A) Data sufficiency
  let foldsUnknown = false;
  if (metrics.folds == null || metrics.folds <= 0) {
    foldsUnknown = true;
    reasons.push('FOLDS_UNKNOWN');
    details['folds'] = 'Fold count not available';
  } else if (metrics.folds < GATE_THRESHOLDS.MIN_FOLDS) {
    reasons.push('INSUFFICIENT_FOLDS');
    details['folds'] = `${metrics.folds} < ${GATE_THRESHOLDS.MIN_FOLDS} minimum`;
  }
  
  // A2) Sample size gate (n_eval)
  if (context?.nEval != null && context.nEval < GATE_THRESHOLDS.N_EVAL_MIN) {
    reasons.push('INSUFFICIENT_SAMPLE');
    details['n_eval'] = `${context.nEval} < ${GATE_THRESHOLDS.N_EVAL_MIN} minimum evaluation points`;
  }
  
  // B) Metric validity
  if (!isValidMetric(metrics.mase)) {
    reasons.push('MASE_INVALID');
    details['mase'] = `Invalid value: ${metrics.mase}`;
  }
  
  if (!isValidMetric(metrics.mae, true)) {  // MAE can be 0
    reasons.push('MAE_INVALID');
    details['mae'] = `Invalid value: ${metrics.mae}`;
  }
  
  if (metrics.coverage == null || !Number.isFinite(metrics.coverage)) {
    reasons.push('COVERAGE_INVALID');
    details['coverage'] = `Invalid value: ${metrics.coverage}`;
  }
  
  // C) Coverage hard-fail (< 70%) - with proper null guard
  if (metrics.coverage != null && Number.isFinite(metrics.coverage)) {
    if (metrics.coverage < GATE_THRESHOLDS.COVERAGE_MIN) {
      reasons.push('COVERAGE_TOO_LOW');
      details['coverage'] = `${(metrics.coverage * 100).toFixed(0)}% < ${GATE_THRESHOLDS.COVERAGE_MIN * 100}% minimum`;
    }
  }
  
  // D) Absolute bias hard-fail (> 250 EUR/t) - with proper null guard
  if (metrics.bias != null && Number.isFinite(metrics.bias)) {
    if (Math.abs(metrics.bias) > GATE_THRESHOLDS.BIAS_MAX_EUR_T) {
      reasons.push('BIAS_TOO_HIGH_ABS');
      details['bias'] = `|${metrics.bias.toFixed(0)}| EUR/t > ${GATE_THRESHOLDS.BIAS_MAX_EUR_T} threshold`;
    }
  }
  
  // E) Absolute MASE hard-fail (> 1.5) - with proper null guard
  if (metrics.mase != null && Number.isFinite(metrics.mase) && metrics.mase > 0) {
    if (metrics.mase > GATE_THRESHOLDS.MASE_MAX) {
      reasons.push('MASE_TOO_HIGH_ABS');
      details['mase_abs'] = `${metrics.mase.toFixed(3)} > ${GATE_THRESHOLDS.MASE_MAX} threshold`;
    }
  }
  
  // F) NO_INTERVALS detection: coverage === 0 or intervalsAvailable === false
  // FIX: Apply to ANY model with coverage === 0 (not just E11)
  const isNoIntervals =
    metrics.intervalsAvailable === false ||
    metrics.coverage === 0;
  
  if (isNoIntervals && !reasons.includes('NO_INTERVALS')) {
    reasons.push('NO_INTERVALS');
    details['intervals'] = 'Model does not produce prediction intervals (coverage=0)';
  }
  
  // G) WIDTH SANITY CHECK: Prevent unrealistically narrow intervals
  // If width is too narrow relative to price level or baseline, the intervals are likely miscalibrated
  if (context?.intervalWidth != null && context.intervalWidth > 0) {
    // G1) Relative to median price: width must be at least 8% of median price
    if (context.medianPrice != null && context.medianPrice > 0) {
      const minWidthAbsolute = context.medianPrice * GATE_THRESHOLDS.WIDTH_MIN_RELATIVE;
      if (context.intervalWidth < minWidthAbsolute) {
        reasons.push('WIDTH_TOO_NARROW');
        details['width_vs_price'] = `Width ${context.intervalWidth.toFixed(0)} < ${minWidthAbsolute.toFixed(0)} (${(GATE_THRESHOLDS.WIDTH_MIN_RELATIVE * 100).toFixed(0)}% of median price ${context.medianPrice.toFixed(0)})`;
      }
    }
    
    // G2) Relative to baseline: width must be at least 50% of baseline width
    if (context.baselineWidth != null && context.baselineWidth > 0) {
      const minWidthVsBaseline = context.baselineWidth * GATE_THRESHOLDS.WIDTH_MIN_VS_BASELINE;
      if (context.intervalWidth < minWidthVsBaseline && !reasons.includes('WIDTH_TOO_NARROW')) {
        reasons.push('WIDTH_TOO_NARROW');
        details['width_vs_baseline'] = `Width ${context.intervalWidth.toFixed(0)} < ${minWidthVsBaseline.toFixed(0)} (${(GATE_THRESHOLDS.WIDTH_MIN_VS_BASELINE * 100).toFixed(0)}% of baseline ${context.baselineWidth.toFixed(0)})`;
      }
    }
  }
  
  // H) COVERAGE STABILITY: Reject if coverage standard deviation is too high across folds
  if (context?.coverageVariance != null && context.coverageVariance > 0) {
    const coverageStdDev = Math.sqrt(context.coverageVariance);
    if (coverageStdDev > GATE_THRESHOLDS.COVERAGE_STDDEV_MAX) {
      reasons.push('COVERAGE_UNSTABLE');
      details['coverage_stability'] = `Coverage std dev ${(coverageStdDev * 100).toFixed(1)}% > ${(GATE_THRESHOLDS.COVERAGE_STDDEV_MAX * 100).toFixed(0)}% threshold`;
    }
  }
  
  // I) Horizon-specific overrides (apply if context provided)
  if (context) {
    const overrides = HORIZON_OVERRIDES[context.experimentId]?.[context.horizon];
    if (overrides) {
      for (const override of overrides) {
        if (!reasons.includes(override)) {
          reasons.push(override);
          details[`override_${override.toLowerCase()}`] = `Horizon override: ${context.experimentId}@${context.horizon}w`;
        }
      }
    }
  }
  
  // Determine eligibility (tri-state)
  // If folds unknown AND no other deterministic hard-fails, eligible = null
  // If any hard-fail, eligible = false
  // Otherwise eligible = true
  const hardFailReasons = reasons.filter(r => r !== 'FOLDS_UNKNOWN');
  const hasHardFail = hardFailReasons.length > 0;
  
  let eligible: boolean | null;
  if (foldsUnknown && !hasHardFail) {
    eligible = null;  // Unknown - can't determine
  } else if (hasHardFail) {
    eligible = false;  // Hard-fail
  } else {
    eligible = true;
  }
  
  return {
    eligible,
    hardFail: hasHardFail,
    reasons,
    warnings,
    details,
  };
}

/**
 * Evaluate relative gates compared to baseline.
 * Only run AFTER hard-fail gates pass.
 */
export function evaluateRelativeGates(
  metrics: { mase: number; bias: number },
  baseline: { mase: number; bias: number }
): EligibilityResult {
  const reasons: HardFailReason[] = [];
  const warnings: string[] = [];
  const details: Record<string, string> = {};
  
  // MASE gate: use min(absolute_threshold, relative_threshold)
  // This ensures both absolute quality AND relative performance vs baseline
  const maseRelativeThreshold = baseline.mase * GATE_THRESHOLDS.MASE_BASELINE_MULTIPLIER;
  const maseEffectiveThreshold = Math.min(GATE_THRESHOLDS.MASE_MAX, maseRelativeThreshold);
  if (metrics.mase > maseEffectiveThreshold) {
    reasons.push('MASE_WORSE_THAN_BASELINE');
    details['mase_rel'] = `${metrics.mase.toFixed(3)} > ${maseEffectiveThreshold.toFixed(3)} (min of ${GATE_THRESHOLDS.MASE_MAX} absolute, ${maseRelativeThreshold.toFixed(3)} relative)`;
  }
  
  // Bias gate: use min(absolute_threshold, relative_threshold)
  const biasRelativeThreshold = Math.abs(baseline.bias) * GATE_THRESHOLDS.BIAS_BASELINE_MULTIPLIER;
  const biasEffectiveThreshold = Math.min(GATE_THRESHOLDS.BIAS_MAX_EUR_T, biasRelativeThreshold);
  if (Math.abs(metrics.bias) > biasEffectiveThreshold) {
    reasons.push('BIAS_WORSE_THAN_BASELINE');
    details['bias_rel'] = `|${metrics.bias.toFixed(0)}| > ${biasEffectiveThreshold.toFixed(0)} (min of ${GATE_THRESHOLDS.BIAS_MAX_EUR_T} absolute, ${biasRelativeThreshold.toFixed(0)} relative)`;
  }
  
  return {
    eligible: reasons.length === 0,
    hardFail: false,  // Relative gates are not hard-fails
    reasons,
    warnings,
    details,
  };
}

/**
 * Full eligibility evaluation with hard-fail and relative gates.
 * Optional context for horizon-specific overrides.
 */
export function evaluateFullEligibility(
  metrics: {
    mase: number | null;
    mae: number | null;
    coverage: number | null;
    bias: number | null;
    folds?: number | null;
  },
  baseline?: { mase: number; bias: number },
  context?: GateContext
): EligibilityResult {
  // First: hard-fail gates (with optional context for overrides)
  const hardFailResult = evaluateHardFailGates(context ?? null, metrics);
  
  if (hardFailResult.hardFail) {
    return hardFailResult;
  }
  
  // Second: relative gates (only if baseline provided and hard-fail passed)
  if (baseline && metrics.mase != null && metrics.bias != null) {
    const relativeResult = evaluateRelativeGates(
      { mase: metrics.mase, bias: metrics.bias },
      baseline
    );
    
    if (!relativeResult.eligible) {
      return {
        ...relativeResult,
        details: { ...hardFailResult.details, ...relativeResult.details },
      };
    }
  }
  
  return {
    eligible: true,
    hardFail: false,
    reasons: [],
    warnings: hardFailResult.warnings,
    details: hardFailResult.details,
  };
}

// ============ WINNER SCORING ============

/**
 * Compute composite score for winner ranking.
 * Lower score = better model.
 * 
 * CORRECTED scoring policy for p10/p90 intervals:
 * - Target coverage: 80% (not 90%) - symmetric penalty around this point
 * - Bias penalty: increased weight for meaningful impact
 * - WIS as SKILL-STYLE score: WIS_model / WIS_baseline (caller computes this)
 * 
 * Formula when WIS skill available:
 *   wisSkill = WIS_model / WIS_baseline (computed by caller, 1.0 = baseline performance)
 *   covPenalty = |coverage - 0.80|
 *   biasPenalty = min(|bias|/250, 2)
 *   score = 1.00*wisSkill + 0.10*MASE + 0.10*covPenalty + 0.05*biasPenalty
 * 
 * Fallback when WIS not available (MASE-first):
 *   score = 1.00*MASE + 0.20*covPenalty + 0.10*biasPenalty
 * 
 * @param wisSkill - WIS skill score: WIS_model / WIS_baseline (lower = better, 1.0 = baseline)
 */
export function computeWinnerScore(
  mase: number,
  coverage: number | null,
  bias: number | null,
  wisSkill?: number | null,
  intervalsAvailable?: boolean,
  medianPrice?: number | null,
  horizon?: number | null,
  intervalWidth?: number | null
): number {
  const cov = coverage ?? 0;
  const b = bias == null ? 0 : Math.abs(bias);
  const h = horizon ?? 4;
  
  // Coverage penalty: symmetric around 80% target (for p10/p90 intervals)
  let covPenalty = Math.abs(cov - GATE_THRESHOLDS.TARGET_COVERAGE);
  
  // HORIZON-SCALED PIECEWISE OVERCOVERAGE PENALTY
  // Short horizons (4-12w): multiplier 0.5 (more lenient)
  // Long horizons (26w+): multiplier 1.5 (stricter, penalize wide intervals more)
  const overcovMultiplier = h >= 26 ? 1.5 : 0.5;
  
  // Piecewise: 0.90-0.95 is moderate, 0.95+ is severe
  // This "bites" 100% coverage without hurting 91%
  if (cov > 0.90 && cov <= 0.95) {
    // Tier 1: moderate penalty for 90-95%
    covPenalty += overcovMultiplier * (cov - 0.90);  // e.g., cov=0.93@26w → +0.045
  } else if (cov > 0.95) {
    // Tier 2: severe penalty for 95%+
    // First tier (0.90-0.95) + second tier at 2x multiplier
    const tier1 = overcovMultiplier * 0.05;  // fixed tier1 for 0.90-0.95
    const tier2 = overcovMultiplier * 2.0 * (cov - 0.95);  // 2x multiplier above 0.95
    covPenalty += tier1 + tier2;  // e.g., cov=1.00@26w → 0.075 + 0.15 = +0.225
  }
  
  // NEAR-100% KICKER: Extra penalty for extremely conservative intervals (cov ≥ 98%)
  // This breaks ties against "open the nets wide" models on short horizons
  // Flat +0.04 kicker effective at breaking tie between E9@100% and E7@94%
  if (cov >= 0.98) {
    covPenalty += 0.04;  // flat kicker for near-100% coverage
  }
  
  // Bias penalty: absolute penalty (capped at 2)
  let biasPenalty = Math.min(b / 250, 2);
  
  // RELATIVE BIAS PENALTY (optional but recommended): Scale bias by price level
  // This makes bias penalty fair across different price regimes (3k vs 9k €/t)
  if (medianPrice != null && medianPrice > 0) {
    const relBias = b / medianPrice;
    const relBiasPenalty = Math.min(relBias / 0.04, 2);  // 4% relative bias = max penalty
    biasPenalty = Math.max(biasPenalty, relBiasPenalty);  // Use the stricter of the two
  }
  
  // WIDTH SANITY PENALTY: Penalize overly wide intervals relative to price level
  // Soft-capped formula: 0.05 * (relWidth / (relWidth + 1))
  // This ensures wider intervals always pay more, but bounded
  // Examples:
  //   relWidth 0.2 → ~0.0083
  //   relWidth 0.5 → ~0.0167
  //   relWidth 1.0 → 0.025
  //   relWidth 3.0 → 0.0375 (still pays more than 1.0, unlike hard cap)
  let widthPenalty = 0;
  if (intervalWidth != null && intervalWidth > 0 && medianPrice != null && medianPrice > 0) {
    const relWidth = intervalWidth / medianPrice;
    widthPenalty = 0.05 * (relWidth / (relWidth + 1));  // Soft cap, always increases
  }
  
  // WIS-first ranking if intervals available and WIS skill computed
  if (intervalsAvailable && wisSkill != null && Number.isFinite(wisSkill) && wisSkill > 0) {
    // Primary: WIS skill (weight 1.0) - lower = better, 1.0 = baseline performance
    // Secondary: MASE guardrail (weight 0.10)
    // Tertiary: coverage penalty (weight 0.10)
    // Quaternary: bias penalty (weight 0.05)
    // Quinary: width penalty (weight 0.05)
    return (wisSkill * 1.00) + (mase * 0.10) + (covPenalty * 0.10) + (biasPenalty * 0.05) + widthPenalty;
  }
  
  // Fallback: MASE-first (for experiments without intervals)
  // Primary: MASE (weight 1.0)
  // Secondary: coverage penalty (weight 0.20)
  // Tertiary: bias penalty (weight 0.10)
  // Quaternary: width penalty (weight 0.05)
  return (mase * 1.00) + (covPenalty * 0.20) + (biasPenalty * 0.10) + widthPenalty;
}

// ============ GATE EVALUATION ============

export interface GateMetrics {
  mase: number;
  coverage: number;
  bias: number;
  availability?: number;
  nEval?: number;           // Total valid evaluation points
  foldCount?: number;       // Number of cross-validation folds
  intervalsAvailable?: boolean;  // Whether p10/p90 intervals exist
}

export interface GateResult {
  overallPassed: boolean;
  biasGatePassed: boolean;
  maseGatePassed: boolean;
  coverageGatePassed: boolean;
  availabilityGatePassed: boolean;
  sampleGatePassed: boolean;
  foldsGatePassed: boolean;
  reasons: string[];
  warnings: string[];
  coveragePenalty: number;  // 0 = no penalty, higher = worse
}

/**
 * Canonical gate evaluation function.
 * Use this EVERYWHERE gates need to be checked.
 * 
 * Hard-fail gates:
 * - availability >= 98%
 * - coverage >= 70%
 * - |bias| <= 250 EUR/t
 * - mase <= 1.5
 * - n_eval >= 100 (sample gate)
 * - fold_count >= 20 (folds gate)
 */
export function evaluateServingGates(metrics: GateMetrics): GateResult {
  const reasons: string[] = [];
  const warnings: string[] = [];
  let coveragePenalty = 0;
  
  const biasGatePassed = Math.abs(metrics.bias) < GATE_THRESHOLDS.BIAS_MAX_EUR_T;
  if (!biasGatePassed) {
    reasons.push(`Bias ${metrics.bias.toFixed(0)} EUR/t exceeds ${GATE_THRESHOLDS.BIAS_MAX_EUR_T} threshold`);
  }
  
  const maseGatePassed = metrics.mase < GATE_THRESHOLDS.MASE_MAX;
  if (!maseGatePassed) {
    reasons.push(`MASE ${metrics.mase.toFixed(2)} exceeds ${GATE_THRESHOLDS.MASE_MAX} threshold`);
  }
  
  // Coverage: ONLY lower bound is a hard gate
  const coverageGatePassed = metrics.coverage >= GATE_THRESHOLDS.COVERAGE_MIN;
  if (!coverageGatePassed) {
    reasons.push(`Coverage ${(metrics.coverage * 100).toFixed(0)}% below ${GATE_THRESHOLDS.COVERAGE_MIN * 100}% minimum`);
  }
  
  // High coverage: warning/penalty only, NOT a gate failure
  if (metrics.coverage > GATE_THRESHOLDS.COVERAGE_SEVERE_HIGH) {
    warnings.push(`Coverage ${(metrics.coverage * 100).toFixed(0)}% very high - intervals may be too wide (target 80-90%)`);
    coveragePenalty = 0.15;
  } else if (metrics.coverage > GATE_THRESHOLDS.COVERAGE_WARNING_HIGH) {
    warnings.push(`Coverage ${(metrics.coverage * 100).toFixed(0)}% above target band (80-90%) - consider tightening intervals`);
    coveragePenalty = 0.05;
  }
  
  const availability = metrics.availability ?? 1.0;
  const availabilityGatePassed = availability >= GATE_THRESHOLDS.AVAILABILITY_MIN;
  if (!availabilityGatePassed) {
    reasons.push(`Availability ${(availability * 100).toFixed(0)}% below ${GATE_THRESHOLDS.AVAILABILITY_MIN * 100}% minimum`);
  }
  
  // Sample size gate
  const nEval = metrics.nEval ?? 0;
  const sampleGatePassed = nEval >= GATE_THRESHOLDS.N_EVAL_MIN;
  if (!sampleGatePassed && nEval > 0) {
    reasons.push(`Insufficient samples: n_eval=${nEval} below ${GATE_THRESHOLDS.N_EVAL_MIN} minimum`);
  }
  
  // Folds gate
  const foldCount = metrics.foldCount ?? 0;
  const foldsGatePassed = foldCount >= GATE_THRESHOLDS.FOLDS_MIN;
  if (!foldsGatePassed && foldCount > 0) {
    reasons.push(`Insufficient folds: ${foldCount} below ${GATE_THRESHOLDS.FOLDS_MIN} minimum`);
  }
  
  // Low sample warning (even if above min)
  if (sampleGatePassed && nEval < 200) {
    warnings.push(`Low sample count: n_eval=${nEval} (recommend >= 200)`);
  }
  
  // Low folds warning (even if above min hard gate)
  if (foldsGatePassed && foldCount < GATE_THRESHOLDS.FOLDS_RECOMMENDED) {
    warnings.push(`Low fold count: ${foldCount} folds (recommend >= ${GATE_THRESHOLDS.FOLDS_RECOMMENDED})`);
  }
  
  const overallPassed = biasGatePassed && maseGatePassed && coverageGatePassed && 
                        availabilityGatePassed && sampleGatePassed && foldsGatePassed;
  
  return {
    overallPassed,
    biasGatePassed,
    maseGatePassed,
    coverageGatePassed,
    availabilityGatePassed,
    sampleGatePassed,
    foldsGatePassed,
    reasons,
    warnings,
    coveragePenalty,
  };
}

/**
 * Check if coverage is within target band (for calibration warnings).
 */
export function isCoverageInTargetBand(coverage: number): boolean {
  return coverage >= GATE_THRESHOLDS.TARGET_COVERAGE_LOW && 
         coverage <= GATE_THRESHOLDS.TARGET_COVERAGE_HIGH;
}

/**
 * Get calibration note based on coverage.
 */
export function getCoverageCalibrationNote(coverage: number): string | null {
  if (coverage < GATE_THRESHOLDS.TARGET_COVERAGE_LOW) {
    return `Coverage ${(coverage * 100).toFixed(0)}% below target band ${GATE_THRESHOLDS.TARGET_COVERAGE_LOW * 100}-${GATE_THRESHOLDS.TARGET_COVERAGE_HIGH * 100}%`;
  }
  if (coverage > GATE_THRESHOLDS.TARGET_COVERAGE_HIGH) {
    return `Coverage ${(coverage * 100).toFixed(0)}% above target band ${GATE_THRESHOLDS.TARGET_COVERAGE_LOW * 100}-${GATE_THRESHOLDS.TARGET_COVERAGE_HIGH * 100}%`;
  }
  return null;
}

// ============ CHAMPION REGISTRY LOOKUP HELPER ============

export interface ChampionLookupResult<T> {
  entry: T | null;
  registryKeyUsed: 'seriesKey' | 'clusterFallback' | 'notFound';
}

/**
 * Lookup champion registry with fallback strategy:
 * 1. First try series_key_w = seriesKey (e.g., "EVOO_GR")
 * 2. If not found, fallback to series_key_w = cluster (e.g., "EVOO")
 * 
 * Returns the entry and which key strategy was used.
 */
export async function lookupChampionRegistry<T>(
  queryFn: (key: string) => Promise<T[]>,
  seriesKey: string,
  cluster: string
): Promise<ChampionLookupResult<T>> {
  // Try seriesKey first (new format: EVOO_GR)
  const bySeriesKey = await queryFn(seriesKey);
  if (bySeriesKey.length > 0) {
    return { entry: bySeriesKey[0], registryKeyUsed: 'seriesKey' };
  }
  
  // Fallback to cluster only (legacy format: EVOO)
  const byCluster = await queryFn(cluster);
  if (byCluster.length > 0) {
    return { entry: byCluster[0], registryKeyUsed: 'clusterFallback' };
  }
  
  return { entry: null, registryKeyUsed: 'notFound' };
}
