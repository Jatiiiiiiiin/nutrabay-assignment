import * as pdfjs from 'pdfjs-dist';

// Configure PDF.js worker
if (typeof window !== 'undefined' && 'Worker' in window) {
  pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.mjs`;
}

/* ================= EMBEDDING PIPELINE ================= */

let embedderPromise: Promise<any> | null = null;

async function getEmbedder() {
  if (!embedderPromise) {
    try {
      const transformersURL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";
      const { pipeline, env } = await import(/* @vite-ignore */ transformersURL);

      env.allowLocalModels = false;
      env.allowRemoteModels = true;
      env.useBrowserCache = true;
      (env as any).isNode = false;
      env.remoteHost = "https://huggingface.co";
      env.remotePath = "";

      embedderPromise = pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    } catch (err) {
      console.error("Neural Engine Initialization Failed:", err);
      embedderPromise = null;
      return null;
    }
  }
  return embedderPromise;
}

/* ================= PDF EXTRACTION (CLIENT) ================= */

export async function extractTextFromPDFClient(file: File | Blob): Promise<string> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjs.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;

    let fullText = "";
    const numPages = Math.min(pdf.numPages, 3);

    for (let i = 1; i <= numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map((item: any) => item.str).join(" ");
      fullText += pageText + " ";
      if (fullText.length > 3000) break;
    }

    return preprocess(fullText).slice(0, 3000);
  } catch (err) {
    console.error("[PDF Client] Extraction failed:", err);
    throw err;
  }
}

/* ================= UTILITIES ================= */

function preprocess(text: unknown): string {
  if (typeof text !== "string") return "";
  return text.trim().replace(/\s+/g, " ").slice(0, 3000);
}

function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    na += vecA[i] * vecA[i];
    nb += vecB[i] * vecB[i];
  }
  const mag = Math.sqrt(na) * Math.sqrt(nb);
  return mag === 0 ? 0 : dot / mag;
}

export function tokenize(text: string): Set<string> {
  const STOP = new Set(["the", "and", "for", "with", "this", "that", "are", "was", "you", "will", "have", "from", "your", "not", "can", "but", "work", "role", "team", "job", "been", "what", "which", "also", "more", "their", "into", "through", "about", "other"]);
  const matches = text.toLowerCase().match(/\b[a-z][a-z0-9+#.\-]{1,}\b/g);
  const words = (matches || []) as string[];
  return new Set(words.filter(w => w.length >= 2 && !STOP.has(w)));
}

function toTitle(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export interface InterviewTips {
  strengths: string[];
  gapAreas: string[];
  powerTips: string[];
}

function buildTips(resumeText: string, jobTitle: string, jobDescription: string): InterviewTips {
  const rTokens = tokenize(resumeText);
  const jTokens = tokenize(jobDescription);

  const matched = [...jTokens].filter(t => rTokens.has(t) && t.length > 3);
  const missing = [...jTokens].filter(t => !rTokens.has(t) && t.length > 3);

  const strengths: string[] = [];
  if (matched.length > 0) strengths.push(`Showcase your hands-on experience with ${matched.slice(0, 3).map(toTitle).join(', ')} — these directly match the JD`);
  if (matched.length > 3) strengths.push(`Demonstrate depth in ${matched.slice(3, 5).map(toTitle).join(' and ')} through specific project examples`);
  if (matched.length > 5) strengths.push(`You share strong overlap in ${matched.slice(5, 7).map(toTitle).join(', ')} — lead with these in technical rounds`);

  const gapAreas: string[] = [];
  if (missing.length > 0) gapAreas.push(`Brush up on ${missing.slice(0, 2).map(toTitle).join(' and ')} — mentioned in the JD but not evident in your resume`);
  if (missing.length > 2) gapAreas.push(`Prepare at least one example or talking point around ${missing.slice(2, 4).map(toTitle).join(', ')}`);

  return { strengths: strengths.slice(0, 3), gapAreas: gapAreas.slice(0, 3), powerTips: [] };
}

const jobEmbeddingCache: Record<string, number[]> = {};

async function computeSemanticScoreLocal(text: string, jobText: string, minRatio: number = 0.03): Promise<number> {
  const embedder = await getEmbedder();
  if (!embedder) return 0;

  const textTokens = tokenize(text);
  const jobTokens = tokenize(jobText);
  if (jobTokens.size === 0) return 0;

  const overlap = [...textTokens].filter(t => jobTokens.has(t)).length;
  const keywordRatio = overlap / jobTokens.size;

  if (keywordRatio < minRatio) return 0;

  const jobHash = jobText.slice(0, 500);
  let jobVec: number[];

  if (jobEmbeddingCache[jobHash]) {
    jobVec = jobEmbeddingCache[jobHash];
  } else {
    const jobOut = await embedder(jobText.slice(0, 2000), { pooling: "mean", normalize: true });
    jobVec = Array.from(jobOut.data || jobOut[0]?.data || jobOut) as number[];
    jobEmbeddingCache[jobHash] = jobVec;
  }

  const textOut = await embedder(text.slice(0, 2000), { pooling: "mean", normalize: true });
  const textVec = Array.from(textOut.data || textOut[0]?.data || textOut) as number[];

  const sim = cosineSimilarity(textVec, jobVec);
  const BASELINE = 0.18, CEILING = 0.75;

  if (sim < BASELINE) return 0;
  const normalized = (sim - BASELINE) / (CEILING - BASELINE);
  return Math.min(1.0, Math.max(0.0, normalized));
}

export async function screenResumeForNutrabay(candidateName: string, resumeText: string, jdText: string) {
  try {
    const rawScore = await computeSemanticScoreLocal(resumeText, jdText, 0.02);
    let matchScore = Math.floor(rawScore * 100);
    
    if (matchScore === 0 && resumeText.length > 50 && jdText.length > 50) {
      const rTokens = tokenize(resumeText);
      const jTokens = tokenize(jdText);
      const overlap = [...rTokens].filter(t => jTokens.has(t)).length;
      matchScore = Math.min(100, Math.floor((overlap / Math.max(10, jTokens.size)) * 100 * 1.5));
    }
    if (matchScore > 100) matchScore = 100;

    let recommendation = "Not Fit";
    if (matchScore >= 80) recommendation = "Strong Fit";
    else if (matchScore >= 50) recommendation = "Moderate Fit";

    const tips = buildTips(resumeText, "Role", jdText);
    
    return {
      candidate: candidateName,
      score: matchScore,
      strengths: tips.strengths.slice(0, 3) || ["Relevant experience found"],
      gaps: tips.gapAreas.slice(0, 3) || ["Missing few keywords"],
      recommendation
    };
  } catch (err) {
    console.error("[Asterix] Nutrabay Screening Failed:", err);
    return { candidate: candidateName, score: 0, strengths: ["Failed to analyze"], gaps: ["Failed to analyze"], recommendation: "Not Fit" };
  }
}