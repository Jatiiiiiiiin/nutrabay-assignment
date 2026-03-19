import React, { useEffect, useState } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import AIResumeScreening from "./views/AIResumeScreening";
import './App.css';

const App: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    let shouldBeDark = saved === "dark" || (!saved && prefersDark);
    document.documentElement.classList.toggle("dark", shouldBeDark);
    setIsDarkMode(shouldBeDark);
  }, []);

  const toggleTheme = () => {
    setIsDarkMode(prev => {
      const next = !prev;
      document.documentElement.classList.toggle("dark", next);
      localStorage.setItem("theme", next ? "dark" : "light");
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-white dark:bg-[#0A0A0A] text-black dark:text-gray-100">
      <Routes>
        <Route path="/" element={<AIResumeScreening onToggleTheme={toggleTheme} isDarkMode={isDarkMode} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
};

export default App;