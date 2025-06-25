"use client";
import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";

// Dynamically import ApexCharts to avoid SSR issues
const Chart = dynamic(() => import("react-apexcharts"), { ssr: false });

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

  const chartOptions = {
    chart: {
      type: 'donut' as const,
    },
    labels: topicData?.labels || [],
    legend: {
      position: 'bottom' as const,
    },
    plotOptions: {
      pie: {
        donut: {
          size: '65%',
        },
      },
    },
    colors: [
      '#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6',
      '#EC4899', '#06B6D4', '#84CC16', '#F97316', '#6366F1'
    ],
  };

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
            Topic Distribution
          </h2>
          
          {topicData && topicData.series.length > 0 ? (
            <div className="flex justify-center">
              <Chart
                options={chartOptions}
                series={topicData.series}
                type="donut"
                width={500}
                height={400}
              />
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