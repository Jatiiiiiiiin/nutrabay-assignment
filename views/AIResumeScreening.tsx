import React, { useState, useRef } from 'react';
import { extractTextFromPDFClient, screenResumeForNutrabay } from '../geminiService';

interface Result {
  candidate: string;
  score: number;
  strengths: string[];
  gaps: string[];
  recommendation: string;
  rank?: number;
}

interface AIResumeScreeningProps {
  onToggleTheme: () => void;
  isDarkMode: boolean;
}

const AIResumeScreening: React.FC<AIResumeScreeningProps> = ({ onToggleTheme, isDarkMode }) => {
  const [jdText, setJdText] = useState("");
  const [resumes, setResumes] = useState<File[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<Result[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setResumes(prev => [...prev, ...Array.from(e.target.files!)]);
    }
  };

  const removeResume = (index: number) => {
    setResumes(prev => prev.filter((_, i) => i !== index));
  };

  const parseAndScore = async () => {
    if (!jdText.trim()) {
      alert("Please paste a Job Description.");
      return;
    }
    if (resumes.length === 0) {
      alert("Please upload at least one resume (PDF).");
      return;
    }

    setIsProcessing(true);
    setResults([]);

    try {
      const parsedResults: Result[] = [];
      
      for (const file of resumes) {
        let resumeText = "";
        try {
          if (file.type === "application/pdf") {
            resumeText = await extractTextFromPDFClient(file);
          } else {
            resumeText = await file.text();
          }
        } catch (e) {
          console.error("Failed to extract:", file.name, e);
          parsedResults.push({
            candidate: file.name,
            score: 0,
            strengths: ["Analysis Failed"],
            gaps: ["Analysis Failed"],
            recommendation: "Not Fit"
          });
          continue;
        }

        const screeningResult = await screenResumeForNutrabay(file.name, resumeText, jdText);
        parsedResults.push(screeningResult);
      }

      // Rank results
      parsedResults.sort((a, b) => b.score - a.score);
      parsedResults.forEach((r, idx) => {
        r.rank = idx + 1;
      });

      setResults(parsedResults);
    } catch (err) {
      console.error("Screening flow failed", err);
      alert("An error occurred during screening.");
    } finally {
      setIsProcessing(false);
    }
  };

  const getRecommendationColor = (rec: string) => {
    if (rec === "Strong Fit") return "text-emerald-500 bg-emerald-500/10 border-emerald-500/20";
    if (rec === "Moderate Fit") return "text-amber-500 bg-amber-500/10 border-amber-500/20";
    return "text-red-500 bg-red-500/10 border-red-500/20";
  };

  return (
    <div className={`min-h-screen bg-[#F0F2F5] dark:bg-[#0A0A0A] text-black dark:text-white font-sans ${isDarkMode ? 'dark' : ''}`}>
      {/* HEADER */}
      <header className="fixed top-0 w-full z-50 bg-white/80 dark:bg-black/80 backdrop-blur-md border-b border-black/5 dark:border-white/5 h-20 flex items-center px-6">
        <div className="flex items-center gap-3">
          <div className="size-10 bg-[#826BF0] rounded-full flex items-center justify-center text-white">
            <span className="material-symbols-outlined text-xl font-black">auto_awesome</span>
          </div>
          <h2 className="text-2xl font-black tracking-tighter leading-none">Asterix JOBs</h2>
          <span className="hidden md:inline-block ml-4 px-3 py-1 rounded-full bg-[#826BF0]/10 text-[#826BF0] text-xs font-bold tracking-widest uppercase">
            AI Resume Screening System
          </span>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <button onClick={onToggleTheme} className="p-2 border border-black/10 dark:border-white/10 rounded-full hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
            <span className="material-symbols-outlined text-sm">{isDarkMode ? 'light_mode' : 'dark_mode'}</span>
          </button>
        </div>
      </header>

      <main className="pt-28 pb-12 px-6 max-w-7xl mx-auto flex flex-col gap-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* JOB DESCRIPTION INPUT */}
          <div className="bg-white dark:bg-[#1A1A1A] rounded-[30px] p-6 lg:p-8 border border-black/5 dark:border-white/5 shadow-xl">
            <h3 className="text-xl font-black uppercase tracking-tighter mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-[#826BF0]">description</span>
              1. Job Description
            </h3>
            <textarea
              className="w-full h-[300px] p-4 rounded-xl bg-gray-50 dark:bg-black/50 border border-black/10 dark:border-white/10 text-sm focus:outline-none focus:border-[#826BF0] transition-colors resize-none"
              placeholder="Paste the target Job Description here..."
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
            />
          </div>

          {/* RESUMES UPLOAD */}
          <div className="bg-white dark:bg-[#1A1A1A] rounded-[30px] p-6 lg:p-8 border border-black/5 dark:border-white/5 shadow-xl flex flex-col">
            <h3 className="text-xl font-black uppercase tracking-tighter mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-[#826BF0]">upload_file</span>
              2. Candidate Resumes (PDF)
            </h3>
            
            <input
              type="file"
              multiple
              accept="application/pdf, text/plain"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileChange}
            />
            
            <button
              onClick={() => fileInputRef.current?.click()}
              className="w-full py-8 border-2 border-dashed border-[#826BF0]/50 rounded-xl hover:bg-[#826BF0]/5 transition-colors flex flex-col items-center justify-center gap-2"
            >
              <span className="material-symbols-outlined text-3xl text-[#826BF0]">add_circle</span>
              <span className="text-sm font-bold text-gray-400">Click to Upload Resumes (5-10 recommended)</span>
            </button>

            <div className="mt-6 flex-grow overflow-y-auto max-h-[160px] space-y-2 pr-2 custom-scrollbar">
              {resumes.map((file, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-black/50 border border-black/5 dark:border-white/5">
                  <div className="flex items-center gap-3 overflow-hidden">
                    <span className="material-symbols-outlined text-red-500 text-lg">picture_as_pdf</span>
                    <span className="text-sm font-medium truncate w-48">{file.name}</span>
                  </div>
                  <button onClick={() => removeResume(idx)} className="text-gray-400 hover:text-red-500 transition-colors">
                    <span className="material-symbols-outlined text-sm">close</span>
                  </button>
                </div>
              ))}
              {resumes.length === 0 && (
                <div className="text-center text-xs text-gray-500 py-4">No resumes scanned yet.</div>
              )}
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <button
            onClick={parseAndScore}
            disabled={isProcessing}
            className={`px-12 py-4 rounded-full text-sm font-black tracking-widest uppercase transition-all shadow-xl flex items-center gap-3 ${
              isProcessing 
              ? 'bg-gray-400 dark:bg-gray-700 cursor-not-allowed opacity-80'
              : 'bg-[#826BF0] hover:bg-[#6c56d6] text-white hover:scale-105 hover:shadow-[#826BF0]/30'
            }`}
          >
            {isProcessing ? (
              <>
                <span className="material-symbols-outlined animate-spin hidden sm:inline-block">refresh</span>
                Neural Core Processing...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined">psychology</span>
                Run AI Screening
              </>
            )}
          </button>
        </div>

        {results.length > 0 && (
          <div className="bg-white dark:bg-[#1A1A1A] rounded-[30px] p-6 lg:p-8 border border-black/5 dark:border-white/5 shadow-xl animate-fade-in-up overflow-x-auto">
            <h3 className="text-xl font-black uppercase tracking-tighter mb-6 flex items-center gap-2">
              <span className="material-symbols-outlined text-[#826BF0]">leaderboard</span>
              Screening Results
            </h3>
            
            <table className="w-full text-left border-collapse min-w-[900px]">
              <thead>
                <tr className="border-b border-black/10 dark:border-white/10">
                  <th className="py-4 px-4 text-xs font-bold uppercase tracking-widest text-gray-500 w-16">Rank</th>
                  <th className="py-4 px-4 text-xs font-bold uppercase tracking-widest text-gray-500 w-1/6">Candidate</th>
                  <th className="py-4 px-4 text-xs font-bold uppercase tracking-widest text-gray-500 w-24">Score</th>
                  <th className="py-4 px-4 text-xs font-bold uppercase tracking-widest text-gray-500 w-1/4">Key Strengths</th>
                  <th className="py-4 px-4 text-xs font-bold uppercase tracking-widest text-gray-500 w-1/4">Key Gaps</th>
                  <th className="py-4 px-4 text-xs font-bold uppercase tracking-widest text-gray-500">Recommendation</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className="border-b border-black/5 dark:border-white/5 hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                    <td className="py-6 px-4">
                      <div className={`size-8 rounded-full flex items-center justify-center font-black text-sm ${r.rank === 1 ? 'bg-yellow-400 text-black shadow-lg shadow-yellow-400/30' : r.rank === 2 ? 'bg-gray-300 text-black' : r.rank === 3 ? 'bg-amber-600 text-white' : 'bg-gray-100 dark:bg-gray-800'}`}>
                        {r.rank}
                      </div>
                    </td>
                    <td className="py-6 px-4 font-bold text-sm break-all">{r.candidate}</td>
                    <td className="py-6 px-4">
                      <div className="flex flex-col gap-1 w-16">
                        <span className="text-xl font-black text-[#826BF0]">{r.score}</span>
                        <div className="h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div className="h-full bg-[#826BF0]" style={{ width: `${r.score}%` }}></div>
                        </div>
                      </div>
                    </td>
                    <td className="py-6 px-4 text-xs text-gray-600 dark:text-gray-300">
                      <ul className="list-disc pl-4 space-y-1">
                        {r.strengths.map((s, idx) => (
                          <li key={idx}>{s}</li>
                        ))}
                      </ul>
                    </td>
                    <td className="py-6 px-4 text-xs text-gray-600 dark:text-gray-300">
                      <ul className="list-disc pl-4 space-y-1">
                        {r.gaps.map((g, idx) => (
                          <li key={idx}>{g}</li>
                        ))}
                      </ul>
                    </td>
                    <td className="py-6 px-4">
                      <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest border inline-block whitespace-nowrap ${getRecommendationColor(r.recommendation)}`}>
                        {r.recommendation}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
      
      <style>{`
        @keyframes fade-in-up {
          0% { opacity: 0; transform: translateY(20px); }
          100% { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
          animation: fade-in-up 0.5s ease-out forwards;
        }
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background-color: rgba(130, 107, 240, 0.3);
          border-radius: 10px;
        }
      `}</style>
    </div>
  );
};

export default AIResumeScreening;
