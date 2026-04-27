/**
 * predictor.js
 * Ensemble generator: combines LSTM and Frequency model outputs.
 */
import { FrequencyModel } from "./frequency-model.js";
import { lstmCandidates } from "./lstm-model.js";
import { GAME_CONFIG } from "./data-ingestion.js";

function validateTicket(ticket, game) {
  const { min, max, pick } = GAME_CONFIG[game];
  return ticket.length === pick && new Set(ticket).size === pick && ticket.every(n => n >= min && n <= max);
}

export async function generatePredictions({ matrix, game = "power655", nTickets = 5, mode = "frequency", lstmModel = null, seqLen = 10, strategy = "all" }) {
  const cfg = GAME_CONFIG[game];
  const fm  = new FrequencyModel(cfg.max, cfg.pick);
  fm.fit(matrix);

  let freqTickets = [], lstmTickets = [];

  if (mode === "frequency" || mode === "ensemble") {
    freqTickets = fm.generateTickets(nTickets, strategy);
  }

  if ((mode === "lstm" || mode === "ensemble") && lstmModel) {
    if (matrix.length >= seqLen) {
      lstmTickets = lstmCandidates(lstmModel, matrix, seqLen, cfg.max, cfg.pick, nTickets);
    }
  }

  const combined = [], seenSets = new Set();
  for (const tkt of [...lstmTickets, ...freqTickets]) {
    const key = tkt.join(",");
    if (!seenSets.has(key) && validateTicket(tkt, game)) {
      seenSets.add(key); combined.push(tkt);
    }
    if (combined.length >= nTickets) break;
  }

  // Top-up if short
  if (combined.length < nTickets) {
    const extras = fm.generateTickets((nTickets - combined.length) * 3, "hot");
    for (const tkt of extras) {
      const key = tkt.join(",");
      if (!seenSets.has(key) && validateTicket(tkt, game)) {
        seenSets.add(key); combined.push(tkt);
      }
      if (combined.length >= nTickets) break;
    }
  }

  return { tickets: combined.slice(0, nTickets), freqModel: fm };
}
