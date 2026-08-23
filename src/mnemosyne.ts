/**
 * Mnemosyne: Zero-Cloud AI Memory Engine on Cloudflare Durable Objects SQLite.
 *
 * Implements the BEAM (Bilevel Episodic-Associative Memory) architecture:
 * 1. Working Memory (hot context, TTL-based short-term storage)
 * 2. Episodic Memory (long-term associative store with 50% Vector + 30% FTS + 20% Importance hybrid recall)
 * 3. Temporal TripleStore (Subject-Predicate-Object Knowledge Graph with validity spans)
 */

export interface MnemosyneConfig {
  enabled: boolean;
  autoRecall: boolean;
  autoRetain: boolean;
  recallTopK: number;
  scope: string; // e.g. "global" or custom domain bank
  updatedAt?: number;
}

export interface WorkingMemoryItem {
  id: string;
  sessionId: string;
  content: string;
  importance: number;
  createdAt: number;
  expiresAt?: number;
}

export interface EpisodicMemoryItem {
  id: string;
  content: string;
  importance: number;
  scope: string;
  source: string;
  embedding?: number[];
  accessCount: number;
  lastAccessed: number;
  createdAt: number;
  validUntil?: number;
  score?: number;
}

export interface TripleItem {
  id: string;
  subject: string;
  predicate: string;
  object: string;
  validFrom?: string;
  validUntil?: string;
  confidence: number;
  source?: string;
  createdAt: number;
}

export interface MnemosyneStats {
  totalEpisodic: number;
  totalWorking: number;
  totalTriples: number;
  totalBanks: number;
  enabled: boolean;
  updatedAt?: number;
}

export interface MnemosyneRecallResult {
  memories: EpisodicMemoryItem[];
  workingMemories: WorkingMemoryItem[];
  triples: TripleItem[];
  formattedContext: string;
}

/* ---------------- Mathematical & Hybrid Search Helpers ---------------- */

export function cosineSimilarity(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length || a.length === 0) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 0;
  return Math.max(0, Math.min(1, (dotProduct / denom + 1) / 2)); // Normalize to 0..1
}

export function tokenizeText(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s_]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

export function tokenOverlapScore(query: string, doc: string): number {
  const qTokens = tokenizeText(query);
  if (qTokens.length === 0) return 0;
  const dLower = doc.toLowerCase();
  let hits = 0;
  for (const t of qTokens) {
    if (dLower.includes(t)) {
      hits++;
    }
  }
  return hits / qTokens.length;
}

export async function generateEmbedding(ai: any, text: string): Promise<number[] | null> {
  if (!ai || !text.trim()) return null;
  try {
    const res = await ai.run("@cf/baai/bge-small-en-v1.5", {
      text: [text.slice(0, 1000)],
    });
    if (res && res.data && Array.isArray(res.data[0])) {
      return res.data[0];
    }
    if (res && Array.isArray(res) && Array.isArray(res[0])) {
      return res[0];
    }
    return null;
  } catch (err) {
    console.warn("Workers AI embedding generation failed (fallback to text matching):", err);
    return null;
  }
}
