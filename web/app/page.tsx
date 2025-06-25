"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const [key, setKey] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  async function run() {
    if (!file) {
      alert("Please select a file");
      return;
    }
    
    if (!key.trim()) {
      alert("Please enter your OpenAI API key");
      return;
    }

    setLoading(true);
    localStorage.setItem("OPENAI_API_KEY", key.trim());
    
    try {
      const formData = new FormData();
      formData.append("file", file);
      
      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData
      });
      
      if (!res.ok) {
        throw new Error(`Upload failed: ${res.statusText}`);
      }
      
      const { job_id } = await res.json();
      router.push(`/dashboard/${job_id}`);
    } catch (error) {
      alert(`Error: ${error}`);
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            ChatGPT Analytics
          </h1>
          <p className="text-gray-600">
            Analyze your ChatGPT conversation data
          </p>
        </div>

        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              OpenAI API Key
            </label>
            <input
              type="password"
              placeholder="sk-..."
              value={key}
              onChange={e => setKey(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              ChatGPT Export File
            </label>
            <input
              type="file"
              accept=".json"
              onChange={e => setFile(e.target.files?.[0] || null)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <p className="text-xs text-gray-500 mt-1">
              Upload your conversations.json file from ChatGPT
            </p>
          </div>

          <button
            onClick={run}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium py-3 px-4 rounded-lg transition-colors"
          >
            {loading ? "Processing..." : "Run Analytics"}
          </button>
        </div>

        <div className="mt-8 text-xs text-gray-500 text-center">
          <p>Your data is processed locally and securely.</p>
          <p>API key is stored only in your browser.</p>
        </div>
      </div>
    </main>
  );
} 