/**
 * data-ingestion.js
 * Parses CSV and JSONL lottery draw files in the browser.
 */

export const GAME_CONFIG = {
  power655: { name: "Power 6/55", min: 1, max: 55, pick: 6, color: "#7c3aed" },
  mega645:  { name: "Mega 6/45",  min: 1, max: 45, pick: 6, color: "#0891b2" },
};

/**
 * Read a File object and return its text content.
 */
export function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = e => resolve(e.target.result);
    reader.onerror = () => reject(new Error("Failed to read file."));
    reader.readAsText(file, "utf-8");
  });
}

/**
 * Parse a JSONL string. Each line: {"id":"…","date":"…","result":[n1…n6]}
 * Returns array of {date, drawId, numbers: [n1..n6]}
 */
export function parseJSONL(text) {
  const records = [];
  const lines = text.split("\n");
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    try {
      const obj = JSON.parse(line);
      const nums = (obj.result || obj.numbers || []).map(Number).sort((a,b)=>a-b);
      if (nums.length < 6) continue;
      records.push({
        date:   obj.date || "",
        drawId: obj.id   || "",
        numbers: nums.slice(0, 6),
      });
    } catch {
      // skip invalid lines
    }
  }
  return sortByDate(records);
}

/**
 * Parse a CSV string. Expected columns: date, draw_id, n1, n2, n3, n4, n5, n6
 * Also handles any CSV with 6 numeric columns (auto-detected).
 */
export function parseCSV(text) {
  const lines = text.split("\n").map(l => l.trim()).filter(Boolean);
  if (lines.length < 2) throw new Error("CSV file appears to be empty.");

  const headers = lines[0].split(",").map(h => h.trim().toLowerCase().replace(/"/g, ""));
  const numCols  = ["n1","n2","n3","n4","n5","n6"];
  const numIdxs  = numCols.map(c => headers.indexOf(c));

  if (numIdxs.some(i => i === -1)) {
    // fallback: pick first 6 numeric-looking columns
    const fallback = headers.reduce((acc, h, i) => {
      if (acc.length < 6) acc.push(i);
      return acc;
    }, []);
    numIdxs.length = 0;
    numIdxs.push(...fallback);
  }

  const dateIdx   = headers.indexOf("date");
  const drawIdIdx = headers.indexOf("draw_id");
  const records   = [];

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",").map(c => c.trim().replace(/"/g, ""));
    const nums = numIdxs.map(idx => parseInt(cols[idx], 10))
                        .filter(n => !isNaN(n))
                        .sort((a,b) => a-b);
    if (nums.length < 6) continue;
    records.push({
      date:   dateIdx   >= 0 ? cols[dateIdx]   : "",
      drawId: drawIdIdx >= 0 ? cols[drawIdIdx] : String(i),
      numbers: nums.slice(0, 6),
    });
  }
  return sortByDate(records);
}

/**
 * Auto-detect format and parse.
 */
export function parseFile(text, filename) {
  const ext = filename.split(".").pop().toLowerCase();
  if (ext === "jsonl" || ext === "json") return parseJSONL(text);
  if (ext === "csv")                     return parseCSV(text);
  // Try JSONL first, then CSV
  try   { return parseJSONL(text); }
  catch { return parseCSV(text);   }
}

function sortByDate(records) {
  return records.sort((a, b) => {
    if (!a.date && !b.date) return 0;
    if (!a.date) return 1;
    if (!b.date) return -1;
    return new Date(a.date) - new Date(b.date);
  });
}

/**
 * Infer game type from max number in dataset.
 */
export function detectGame(records) {
  const maxNum = Math.max(...records.flatMap(r => r.numbers));
  return maxNum > 45 ? "power655" : "mega645";
}

/**
 * Extract raw (n_draws × 6) integer matrix.
 */
export function getMatrix(records) {
  return records.map(r => r.numbers);
}
