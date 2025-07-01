"use client";
import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";

interface JobStatus {
  progress: number;
  ready: boolean;
  error?: string;
  message_count?: number;
}

interface TopicData {
  series: number[];
  labels: string[];
  message_count: number;
}

export default function Dashboard() {
  const params = useParams();
  const router = useRouter();
  const jobId = params.job_id as string;
  
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [topicData, setTopicData] = useState<TopicData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const pollStatus = async () => {
      try {
        const res = await fetch(`http://127.0.0.1:8000/status/${jobId}`);
        const data = await res.json();
        
        if (data.error) {
          setError(data.error);
          return;
        }
        
        setStatus(data);
        
        if (data.ready && !data.error) {
          // Fetch topic data
          const topicRes = await fetch(`http://127.0.0.1:8000/topics/${jobId}`);
          const topicData = await topicRes.json();
          
          if (topicData.error) {
            setError(topicData.error);
          } else {
            setTopicData(topicData);
          }
        }
      } catch (err) {
        setError("Failed to fetch status");
      }
    };

    // Poll every 2 seconds until ready
    const interval = setInterval(pollStatus, 2000);
    pollStatus(); // Initial call

    return () => clearInterval(interval);
  }, [jobId]);

  if (error) {
    return (
      <div className="min-h-screen bg-red-50 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md text-center">
          <div className="text-red-600 text-xl mb-4">Error</div>
          <p className="text-gray-700 mb-4">{error}</p>
          <button
            onClick={() => router.push("/")}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!status || !status.ready) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-xl p-8 w-full max-w-md text-center">
          <div className="mb-6">
            <div className="text-2xl font-bold text-gray-900 mb-2">
              Processing Your Data
            </div>
            <p className="text-gray-600">
              Analyzing your ChatGPT conversations...
            </p>
          </div>
          
          <div className="mb-6">
            <div className="bg-gray-200 rounded-full h-3 mb-2">
              <div 
                className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                style={{ width: `${status?.progress || 0}%` }}
              />
            </div>
            <div className="text-sm text-gray-600">
              {status?.progress || 0}% Complete
            </div>
          </div>
          
          <div className="animate-spin mx-auto w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full" />
        </div>
      </div>
    );
  }



  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Your ChatGPT Analytics
          </h1>
          <p className="text-gray-600">
            Analysis of {topicData?.message_count || 0} messages
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6 text-center">
            Top Conversation Topics
          </h2>
          
          {topicData && topicData.series.length > 0 ? (
            <div className="space-y-3">
              {topicData.labels.map((label, index) => {
                const count = topicData.series[index];
                const totalConversations = topicData.series.reduce((a, b) => a + b, 0);
                const percentage = ((count / totalConversations) * 100).toFixed(1);
                
                return (
                  <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                    <div className="flex-1">
                      <div className="font-medium text-gray-900 mb-1">
                        {label}
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                    <div className="ml-4 text-right">
                      <div className="text-lg font-semibold text-gray-900">
                        {count}
                      </div>
                      <div className="text-sm text-gray-500">
                        {percentage}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              No topics found in your data
            </div>
          )}
        </div>

        <div className="text-center mt-8">
          <button
            onClick={() => router.push("/")}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg"
          >
            Analyze Another File
          </button>
        </div>
      </div>
    </div>
  );
} 