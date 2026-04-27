/**
 * lstm-model.js
 * TensorFlow.js LSTM model for Vietlott sequence prediction.
 */

export function buildLSTMModel(seqLen, nFeatures = 6) {
  const model = tf.sequential({ name: "vietlott_lstm" });
  model.add(tf.layers.lstm({ units: 64, inputShape: [seqLen, nFeatures], returnSequences: true }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.lstm({ units: 32, returnSequences: false }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  model.add(tf.layers.dense({ units: nFeatures, activation: "sigmoid" }));
  model.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError", metrics: ["mae"] });
  return model;
}

export function buildSequences(matrix, seqLen, maxVal) {
  const normed = matrix.map(row => row.map(n => n / maxVal));
  const xData = [], yData = [];
  for (let i = 0; i < normed.length - seqLen; i++) {
    xData.push(normed.slice(i, i + seqLen));
    yData.push(normed[i + seqLen]);
  }
  if (xData.length === 0) throw new Error(`Not enough draws (${matrix.length}) for seq_len=${seqLen}.`);
  return { xs: tf.tensor3d(xData), ys: tf.tensor2d(yData), count: xData.length };
}

export async function trainLSTM(model, xs, ys, opts = {}, onEpochEnd = null) {
  const { epochs = 60, batchSize = 32, validationSplit = 0.1 } = opts;
  let bestVal = Infinity, patience = 0;
  const lossH = [], valH = [];
  return model.fit(xs, ys, {
    epochs, batchSize, validationSplit, shuffle: true,
    callbacks: {
      onEpochEnd: async (ep, logs) => {
        lossH.push(logs.loss); valH.push(logs.val_loss);
        if (onEpochEnd) onEpochEnd(ep + 1, epochs, logs, lossH, valH);
        if (logs.val_loss < bestVal - 1e-5) { bestVal = logs.val_loss; patience = 0; }
        else if (++patience >= 12) model.stopTraining = true;
        await tf.nextFrame();
      }
    }
  });
}

export function predictRaw(model, lastSeq, maxVal) {
  return tf.tidy(() => {
    const normed = lastSeq.map(row => row.map(n => n / maxVal));
    const out = model.predict(tf.tensor3d([normed])).arraySync()[0];
    return out.map(v => v * maxVal);
  });
}

export function lstmCandidates(model, matrix, seqLen, maxVal, pick, n = 5) {
  const tickets = [], seenSets = new Set();
  for (let attempt = 0; attempt < n * 40 && tickets.length < n; attempt++) {
    const baseSeq = matrix.slice(-seqLen);
    const noisySeq = baseSeq.map(row => row.map(n => Math.max(1, Math.min(maxVal, n + (Math.random() - 0.5) * 2))));
    const raw = predictRaw(model, noisySeq, maxVal);
    const cands = raw.map(v => Math.round(Math.max(1, Math.min(maxVal, v))));
    const seen = new Set(), resolved = [];
    for (const c of cands) {
      let b = c, t = 0;
      while (seen.has(b) && t++ < 20) b = Math.max(1, Math.min(maxVal, b + (Math.random() > 0.5 ? 1 : -1)));
      if (!seen.has(b)) { resolved.push(b); seen.add(b); }
    }
    if (resolved.length < pick) continue;
    const tkt = resolved.slice(0, pick).sort((a, b) => a - b);
    const key = tkt.join(",");
    if (!seenSets.has(key) && tkt.every(b => b >= 1 && b <= maxVal)) { seenSets.add(key); tickets.push(tkt); }
  }
  return tickets;
}
