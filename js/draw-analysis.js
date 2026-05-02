/**
 * draw-analysis.js — Deep Analysis Report for Last N Draws
 * =========================================================
 * Generates a fully-detailed statistical breakdown of why each number
 * appeared in the last 10 draws, with percentages, context, and narrative.
 */

/**
 * Analyze the last N draws in detail against the full historical dataset.
 *
 * @param {Array}          records   – all historical records [{date, numbers}, ...]
 * @param {Object}         fm        – fitted FrequencyModel instance
 * @param {Object}         gameCfg   – { max, pick, name }
 * @param {number}         [n=10]    – number of recent draws to analyze
 * @returns {Object}                  – structured analysis report
 */
export function analyzeRecentDraws(records, fm, gameCfg, n = 10) {
  if (!records?.length || !fm?.fitted) return null;

  const totalDraws  = records.length;
  const recentSlice = records.slice(-n);           // last N draws (oldest → newest)
  const maxVal      = gameCfg.max;
  const pick        = gameCfg.pick;

  // ── All numbers that appeared in recent window ──────────────────────────────
  const ballStats = {}; // ball → { appearances, draws, ... }
  const allNums   = new Set();

  recentSlice.forEach((rec, idx) => {
    rec.numbers.forEach(b => {
      allNums.add(b);
      if (!ballStats[b]) {
        ballStats[b] = {
          ball:        b,
          appearances: 0,   // count in last N draws
          drawIdxs:    [],  // which draw indices (0-based within slice)
        };
      }
      ballStats[b].appearances++;
      ballStats[b].drawIdxs.push(idx);
    });
  });

  // ── Per-ball enrichment ─────────────────────────────────────────────────────
  const enriched = [...allNums].map(b => {
    const s            = ballStats[b];
    const appearances  = s.appearances;
    const appearedDraws = s.drawIdxs;

    // Frequency metrics
    const allTimePct   = ((fm.counts[b] / totalDraws) * 100).toFixed(1);
    const recentPct    = ((appearances / n) * 100).toFixed(1);
    const expectedPct  = ((pick / maxVal) * 100).toFixed(1);
    const allTimePctN  = (fm.counts[b] / totalDraws) * 100;
    const recentPctN   = (appearances / n) * 100;
    const expectedPctN = (pick / maxVal) * 100;

    // Deviation from expected
    const deviation    = (recentPctN - expectedPctN).toFixed(1);
    const deviationAbs = Math.abs(recentPctN - expectedPctN);

    // Bayesian / model signals
    const bayesP       = (fm.bayesianP(b) * 100).toFixed(2);
    const blendedP     = (fm.blendedP(b)  * 100).toFixed(2);
    const trendScore   = fm.trend(b);
    const gapNow       = fm.rawGaps?.[b] ?? 0;
    const gapZ         = fm.gapZ(b).toFixed(2);
    const freqZ        = fm.freqZ(b).toFixed(2);
    const isSig        = fm.isSignificant(b);

    // Classification
    const hotSet     = new Set(fm.hotNumbers);
    const coldSet    = new Set(fm.coldNumbers);
    const isHot      = hotSet.has(b);
    const isCold     = coldSet.has(b);
    const isOverdue  = gapNow >= Math.round(fm.muGap + fm.sdGap);  // gap ≥ mean + 1 SD

    // Trend classification
    let trendLabel, trendClass;
    if (trendScore > 0.04)       { trendLabel = "📈 Đang tăng mạnh";  trendClass = "rising-strong"; }
    else if (trendScore > 0.015) { trendLabel = "⬆ Tăng nhẹ";        trendClass = "rising"; }
    else if (trendScore < -0.04) { trendLabel = "📉 Đang giảm mạnh"; trendClass = "falling-strong"; }
    else if (trendScore < -0.015){ trendLabel = "⬇ Giảm nhẹ";        trendClass = "falling"; }
    else                         { trendLabel = "➡ Ổn định";          trendClass = "stable"; }

    // Ball type label
    let typeLabel, typeClass;
    if (isHot && trendScore > 0)       { typeLabel = "🔥 Nóng & Tăng";   typeClass = "hot-rising"; }
    else if (isHot)                    { typeLabel = "🔥 Nóng";           typeClass = "hot"; }
    else if (isCold && trendScore < 0) { typeLabel = "🧊 Lạnh & Giảm";   typeClass = "cold-falling"; }
    else if (isCold)                   { typeLabel = "🧊 Lạnh";           typeClass = "cold"; }
    else if (isOverdue)                { typeLabel = "⏳ Quá hạn";        typeClass = "overdue"; }
    else                               { typeLabel = "⚪ Trung tính";      typeClass = "neutral"; }

    // Narrative reason for appearance
    const reasons = _buildReasons({
      appearances, n, expectedPctN, recentPctN, allTimePctN,
      isHot, isCold, isOverdue, isSig, trendScore, gapNow, freqZ: parseFloat(freqZ),
    });

    // Best pair partner (highest lift with this ball in ALL history)
    let bestPairBall = null, bestPairLift = 0;
    for (let j = 1; j <= maxVal; j++) {
      if (j === b) continue;
      const lift = fm.pairLift(b, j);
      if (lift > bestPairLift) { bestPairLift = lift; bestPairBall = j; }
    }

    return {
      ball:         b,
      appearances,
      appearedIn:   appearedDraws,
      recentPct:    parseFloat(recentPct),
      allTimePct:   parseFloat(allTimePct),
      expectedPct:  parseFloat(expectedPct),
      deviation:    parseFloat(deviation),
      deviationAbs,
      bayesP:       parseFloat(bayesP),
      blendedP:     parseFloat(blendedP),
      trendScore:   trendScore.toFixed(4),
      trendLabel,
      trendClass,
      typeLabel,
      typeClass,
      gapNow,
      gapZ:         parseFloat(gapZ),
      freqZ:        parseFloat(freqZ),
      isSig,
      isHot,
      isCold,
      isOverdue,
      allTimeCount: fm.counts[b],
      reasons,
      bestPairBall,
      bestPairLift: bestPairLift.toFixed(2),
    };
  });

  // Sort by appearances desc, then by allTimePct desc
  enriched.sort((a, b) => b.appearances - a.appearances || b.allTimePct - a.allTimePct);

  // ── Draw-level summaries ─────────────────────────────────────────────────────
  const hotSet  = new Set(fm.hotNumbers);
  const coldSet = new Set(fm.coldNumbers);

  const drawSummaries = recentSlice.map((rec, idx) => {
    const nums       = rec.numbers;
    const hotCount   = nums.filter(b => hotSet.has(b)).length;
    const coldCount  = nums.filter(b => coldSet.has(b)).length;
    const risingCount = nums.filter(b => (fm.trendScores?.[b] ?? 0) > 0.015).length;
    const pairLifts  = [];
    for (let i = 0; i < nums.length; i++) {
      for (let j = i + 1; j < nums.length; j++) {
        pairLifts.push({ a: nums[i], b: nums[j], lift: fm.pairLift(nums[i], nums[j]) });
      }
    }
    const topPair = pairLifts.sort((a, b) => b.lift - a.lift)[0];

    return {
      idx:          totalDraws - n + idx + 1,  // kỳ number
      date:         rec.date?.toString().slice(0, 10) ?? "—",
      numbers:      nums,
      hotCount,
      coldCount,
      risingCount,
      topPair,
      drawCharacter: _drawCharacter(hotCount, coldCount, risingCount, pick),
    };
  });

  // ── Pair co-occurrence matrix for recent draws ───────────────────────────────
  const recentPairCounts = new Map();
  recentSlice.forEach(rec => {
    const nums = [...rec.numbers].sort((a, b) => a - b);
    for (let i = 0; i < nums.length; i++) {
      for (let j = i + 1; j < nums.length; j++) {
        const key = `${nums[i]},${nums[j]}`;
        recentPairCounts.set(key, (recentPairCounts.get(key) || 0) + 1);
      }
    }
  });

  // Top pairs that appeared 2+ times in recent window
  const repeatPairs = [...recentPairCounts.entries()]
    .filter(([, cnt]) => cnt >= 2)
    .map(([key, cnt]) => {
      const [a, b] = key.split(",").map(Number);
      return { a, b, cnt, lift: fm.pairLift(a, b) };
    })
    .sort((a, b) => b.cnt - a.cnt || b.lift - a.lift)
    .slice(0, 10);

  // ── Summary statistics ───────────────────────────────────────────────────────
  const uniqueBalls     = allNums.size;
  const totalBallSlots  = n * pick;
  const coverageRatio   = ((uniqueBalls / maxVal) * 100).toFixed(1);
  const avgAppearances  = (totalBallSlots / uniqueBalls).toFixed(2);

  const hotAppeared   = enriched.filter(e => e.isHot).length;
  const coldAppeared  = enriched.filter(e => e.isCold).length;
  const overdueApp    = enriched.filter(e => e.isOverdue).length;
  const sigBalls      = enriched.filter(e => e.isSig);

  return {
    meta: {
      totalDraws,
      analyzedDraws: n,
      uniqueBalls,
      totalBallSlots,
      coverageRatio: parseFloat(coverageRatio),
      avgAppearances: parseFloat(avgAppearances),
      gameName: gameCfg.name,
      maxVal,
      pick,
      expectedPct: ((pick / maxVal) * 100).toFixed(1),
      muGap: fm.muGap?.toFixed(1) ?? "9.2",
      sdGap: fm.sdGap?.toFixed(1) ?? "8.7",
    },
    summary: {
      hotAppeared,
      coldAppeared,
      overdueApp,
      sigBalls: sigBalls.map(e => e.ball),
    },
    enriched,       // per-ball detail
    drawSummaries,  // per-draw summary (oldest → newest)
    repeatPairs,    // pairs appearing 2+ times
  };
}

// ── Internal helpers ───────────────────────────────────────────────────────────

function _buildReasons({ appearances, n, expectedPctN, recentPctN, allTimePctN,
                         isHot, isCold, isOverdue, isSig, trendScore, gapNow, freqZ }) {
  const reasons = [];
  const overRatio = recentPctN / expectedPctN;

  if (appearances >= 3) {
    reasons.push(`Xuất hiện ${appearances}/${n} kỳ (rất nhiều so với kỳ vọng ${expectedPctN.toFixed(0)}%)`);
  } else if (appearances === 2) {
    reasons.push(`Xuất hiện 2 lần trong ${n} kỳ (gấp đôi kỳ vọng)`);
  } else {
    reasons.push(`Xuất hiện 1 lần trong ${n} kỳ (đúng với xác suất kỳ vọng)`);
  }

  if (isHot) {
    reasons.push(`Thuộc nhóm HÓT lịch sử (lịch sử xuất hiện ${allTimePctN.toFixed(1)}% > kỳ vọng ${expectedPctN.toFixed(0)}%)`);
  }
  if (isCold) {
    reasons.push(`Thuộc nhóm LẠNH (ít xuất hiện trong lịch sử, ${allTimePctN.toFixed(1)}%)`);
  }
  if (isOverdue && gapNow > 0) {
    reasons.push(`Đã vắng mặt ${gapNow} kỳ liên tiếp trước khi xuất hiện (khoảng cách bình thường: ~${Math.round(expectedPctN === 0 ? 9 : 100/expectedPctN)} kỳ)`);
  }
  if (trendScore > 0.03) {
    reasons.push(`Xu hướng TĂNG mạnh trong 50 kỳ gần nhất (+${(trendScore * 100).toFixed(1)}% hơn trung bình)`);
  } else if (trendScore > 0.01) {
    reasons.push(`Xu hướng tăng nhẹ gần đây (+${(trendScore * 100).toFixed(1)}%)`);
  } else if (trendScore < -0.03) {
    reasons.push(`Xu hướng GIẢM trong 50 kỳ gần nhất (${(trendScore * 100).toFixed(1)}% thấp hơn trung bình)`);
  }
  if (isSig && freqZ > 0) {
    reasons.push(`Tần suất lịch sử CAO có ý nghĩa thống kê (Z = ${freqZ.toFixed(1)}, vượt ngưỡng Bonferroni)`);
  } else if (isSig && freqZ < 0) {
    reasons.push(`Tần suất lịch sử THẤP có ý nghĩa thống kê (Z = ${freqZ.toFixed(1)})`);
  }

  return reasons;
}

function _drawCharacter(hotCount, coldCount, risingCount, pick) {
  if (hotCount >= 4)   return { label: "🔥 Thiên về số HOT",   cls: "char-hot" };
  if (coldCount >= 4)  return { label: "🧊 Thiên về số LẠNH",  cls: "char-cold" };
  if (risingCount >= 3)return { label: "📈 Xu hướng TĂNG",     cls: "char-rising" };
  if (hotCount >= 2 && coldCount >= 2) return { label: "⚖️ Cân bằng HOT/LẠNH", cls: "char-balanced" };
  return { label: "⚪ Hỗn hợp",                                  cls: "char-mixed" };
}
