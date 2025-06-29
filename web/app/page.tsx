"use client";
import React, { useState, useEffect } from 'react';
import { Upload, Terminal, Zap, Moon, Sun, MessageSquare, Database, Clock, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useTheme } from 'next-themes';
import dynamic from 'next/dynamic';

// Dynamically import ApexCharts with better error handling
const Chart = dynamic(() => import('react-apexcharts'), { 
  ssr: false,
  loading: () => <div className="flex items-center justify-center h-64"><Loader2 className="w-8 h-8 animate-spin text-terminal" /></div>
});

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, selectedFile }) => {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type === 'application/json') {
        onFileSelect(file);
      } else {
        alert('ERROR_INVALID_FILE: JSON file required for ChatGPT conversations');
        event.target.value = '';
      }
    }
  };

  return (
    <div className="space-y-2">
      <Label htmlFor="file" className="text-sm font-mono font-medium">
        CHATGPT_CONVERSATIONS.JSON:
      </Label>
      <div className="relative">
        {/* Hidden file input */}
        <input
          id="file"
          type="file"
          accept=".json"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
        />
        
        {/* Custom styled overlay */}
        <div className="font-mono retro-border bg-muted/30 h-10 px-3 py-2 flex items-center relative">
          <div className="flex items-center gap-3">
            <div className="bg-terminal text-black px-3 py-1 rounded-md font-bold text-sm">
              Choose File
            </div>
            <span className="text-muted-foreground text-sm">
              {selectedFile ? selectedFile.name : 'No file chosen'}
            </span>
          </div>
          <Upload className="absolute right-3 w-4 h-4 text-muted-foreground pointer-events-none" />
        </div>
      </div>
      {selectedFile && (
        <p className="text-xs text-terminal font-mono text-center">
          {'>'} {selectedFile.name} [LOADED]
        </p>
      )}
      <p className="text-xs text-muted-foreground font-mono">
        // export from ChatGPT settings → data controls
      </p>
    </div>
  );
};

// New separate loading screen component
const LoadingScreen: React.FC<{ progress: number; message: string; theme: string | undefined }> = ({ progress, message, theme }) => {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      {/* Theme toggle */}
      <div className="absolute top-4 right-4 z-20">
        <Button
          variant="outline"
          size="sm"
          onClick={() => {}} // Disabled during loading
          disabled
          className="retro-border terminal-glow font-mono opacity-50"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </Button>
      </div>

      <div className="w-full max-w-md sm:max-w-lg px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Terminal className="w-8 h-8 text-terminal mr-3" />
            <h1 className="text-2xl sm:text-3xl font-bold text-terminal font-mono">
              GPT_ANALYTICS
            </h1>
          </div>
          <p className="text-muted-foreground font-mono text-sm">
            {'>'} PROCESSING_DATA_STREAM...
          </p>
        </div>

        {/* Loading Card */}
        <Card className="retro-border bg-card terminal-glow">
          <CardHeader className="text-center border-b border-border">
            <CardTitle className="text-xl flex items-center justify-center gap-2 font-mono">
              <Loader2 className="w-5 h-5 text-terminal animate-spin" />
              ANALYZING_CONVERSATIONS
            </CardTitle>
            <CardDescription className="font-mono text-muted-foreground">
              neural network processing active
            </CardDescription>
          </CardHeader>

          <CardContent className="p-6">
            {/* Processing steps */}
            <div className="space-y-3 text-sm font-mono">
              <div className={`flex items-center justify-between ${progress > 0 ? 'text-terminal' : 'text-muted-foreground'}`}>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-current"></div>
                  INITIALIZING_ANALYSIS
                </div>
                <div className="flex items-center">
                  {progress > 0 ? (
                    <span className="text-xs opacity-70">✓</span>
                  ) : progress === 0 && (
                    <Loader2 className="w-3 h-3 animate-spin opacity-50" />
                  )}
                </div>
              </div>
              
              <div className={`flex items-center justify-between ${progress > 10 ? 'text-terminal' : 'text-muted-foreground'}`}>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-current"></div>
                  PARSING_JSON_DATA
                </div>
                <div className="flex items-center">
                  {progress > 10 ? (
                    <span className="text-xs opacity-70">✓</span>
                  ) : progress > 0 && progress <= 10 && (
                    <Loader2 className="w-3 h-3 animate-spin opacity-50" />
                  )}
                </div>
              </div>
              
              <div className={`flex items-center justify-between ${progress > 30 ? 'text-terminal' : 'text-muted-foreground'}`}>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-current"></div>
                  EXTRACTING_CONVERSATIONS
                </div>
                <div className="flex items-center">
                  {progress > 30 ? (
                    <span className="text-xs opacity-70">✓</span>
                  ) : progress > 10 && progress <= 30 && (
                    <Loader2 className="w-3 h-3 animate-spin opacity-50" />
                  )}
                </div>
              </div>
              
              <div className={`flex items-center justify-between ${progress > 60 ? 'text-terminal' : 'text-muted-foreground'}`}>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-current"></div>
                  RUNNING_TOPIC_ANALYSIS
                </div>
                <div className="flex items-center">
                  {progress > 60 ? (
                    <span className="text-xs opacity-70">✓</span>
                  ) : progress > 30 && progress <= 60 && (
                    <Loader2 className="w-3 h-3 animate-spin opacity-50" />
                  )}
                </div>
              </div>
              
              <div className={`flex items-center justify-between ${progress > 90 ? 'text-terminal' : 'text-muted-foreground'}`}>
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-current"></div>
                  GENERATING_INSIGHTS
                </div>
                <div className="flex items-center">
                  {progress > 90 ? (
                    <span className="text-xs opacity-70">✓</span>
                  ) : progress > 60 && progress <= 90 && (
                    <Loader2 className="w-3 h-3 animate-spin opacity-50" />
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

interface ProgressProps {
  progress: number;
  message: string;
  visible: boolean;
}

const Progress: React.FC<ProgressProps> = ({ progress, message, visible }) => {
  if (!visible) return null;

  return (
    <div className="mt-6 p-6 bg-muted/30 retro-border rounded-md">
      <div className="w-full h-2 bg-muted rounded-full overflow-hidden mb-4">
        <div
          className="h-full bg-terminal transition-all duration-500 relative"
          style={{ width: `${progress}%` }}
        >
          <div className="absolute inset-0 shimmer rounded-full" />
        </div>
      </div>
      <div className="text-center text-sm text-muted-foreground font-mono font-medium">
        {message}
      </div>
    </div>
  );
};

interface StatCardProps {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  value: number;
  description: string;
}

const StatCard: React.FC<StatCardProps> = ({ icon: Icon, title, value, description }) => (
  <Card className="retro-border bg-card terminal-glow hover:scale-105 transition-transform duration-200">
    <CardHeader className="pb-2">
      <CardTitle className="text-sm font-mono font-medium flex items-center gap-2">
        <Icon className="w-4 h-4 text-terminal" />
        {title}
      </CardTitle>
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold text-terminal font-mono">
        {value.toLocaleString()}
      </div>
      <p className="text-xs text-muted-foreground mt-1 font-mono">{description}</p>
    </CardContent>
  </Card>
);

interface ChartCardProps {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}

const ChartCard: React.FC<ChartCardProps> = ({ title, description, icon: Icon, children }) => (
  <Card className="retro-border bg-card terminal-glow">
    <CardHeader className="border-b border-border">
      <CardTitle className="text-lg flex items-center gap-2 font-mono">
        <Icon className="w-5 h-5 text-terminal" />
        {title}
      </CardTitle>
      <CardDescription className="font-mono text-muted-foreground">
        {description}
      </CardDescription>
    </CardHeader>
    <CardContent className="p-6">
      {children}
    </CardContent>
  </Card>
);

export default function Home() {
  const [apiKey, setApiKey] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const showError = (message: string) => {
    setError(message);
    setTimeout(() => setError(null), 5000);
  };

  const pollStatus = async (jobId: string) => {
    try {
      console.log(`DEBUG: Polling status for job ${jobId}`);
      const response = await fetch(`http://127.0.0.1:8000/status/${jobId}`);
      const status = await response.json();
      console.log(`DEBUG: Status response:`, status);

      if (status.error) {
        console.log(`DEBUG: Error in status:`, status.error);
        showError(`PROCESSING_ERROR: ${status.error}`);
        setIsLoading(false);
        return;
      }

      console.log(`DEBUG: Progress: ${status.progress}, Ready: ${status.ready}`);
      setProgress(status.progress || 0);

      if (status.ready) {
        console.log(`DEBUG: Job ready, loading results...`);
        await loadResults(jobId);
      } else {
        console.log(`DEBUG: Job not ready, polling again in 2s...`);
        setTimeout(() => pollStatus(jobId), 2000);
      }
    } catch (error) {
      console.log(`DEBUG: Polling error:`, error);
      showError('STATUS_CHECK_FAILED: Unable to check processing status');
      setIsLoading(false);
    }
  };

  const loadResults = async (jobId: string) => {
    try {
      const [topicsResponse, modelsResponse, dailyResponse] = await Promise.all([
        fetch(`http://127.0.0.1:8000/topics/${jobId}`),
        fetch(`http://127.0.0.1:8000/models/${jobId}`),
        fetch(`http://127.0.0.1:8000/daily/${jobId}`)
      ]);

      const topicsData = await topicsResponse.json();
      const modelsData = await modelsResponse.json();
      const dailyData = await dailyResponse.json();

      if (topicsData.error || modelsData.error || dailyData.error) {
        showError(`DATA_LOAD_ERROR: ${topicsData.error || modelsData.error || dailyData.error}`);
        return;
      }

      setResults({ topics: topicsData, models: modelsData, daily: dailyData });
      setIsLoading(false);
    } catch (error) {
      showError('RESULT_LOAD_FAILED: Unable to load analysis results');
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!selectedFile) {
      showError('FILE_REQUIRED: Please select a ChatGPT conversations file');
      return;
    }

    console.log(`DEBUG: Starting upload for file:`, selectedFile.name);
    setIsLoading(true);
    setProgress(0);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      // Only include API key if provided
      if (apiKey.trim()) {
        console.log(`DEBUG: Including API key in upload`);
        formData.append('api_key', apiKey);
      } else {
        console.log(`DEBUG: No API key provided, using simple analysis`);
      }

      console.log(`DEBUG: Uploading to backend...`);
      const response = await fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      console.log(`DEBUG: Upload response:`, result);

      if (result.error) {
        console.log(`DEBUG: Upload error:`, result.error);
        showError(`UPLOAD_ERROR: ${result.error}`);
        setIsLoading(false);
        return;
      }

      console.log(`DEBUG: Upload successful, job ID: ${result.job_id}`);
      setCurrentJobId(result.job_id);
      pollStatus(result.job_id);
    } catch (error) {
      console.log(`DEBUG: Upload failed:`, error);
      showError('UPLOAD_FAILED: Unable to upload file for analysis');
      setIsLoading(false);
    }
  };

  const createTopicsChart = (data: any) => {
    if (!data.series || data.series.length === 0) {
      return <p className="text-center text-muted-foreground text-sm font-mono">No topics found in conversations</p>;
    }

    // Responsive chart height based on screen size - LARGER for topics chart to match models section total height
    const getChartHeight = () => {
      if (typeof window !== 'undefined') {
        return window.innerWidth < 640 ? 400 : window.innerWidth < 1024 ? 500 : 600;
      }
      return 500;
    };

    // More robust theme detection with fallback
    const isDarkMode = theme === 'dark' || (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    const legendTextColor = isDarkMode ? '#ffffff' : '#000000'; // Pure white/black for maximum contrast
    const dataLabelColor = isDarkMode ? '#ffffff' : '#000000';
    
    console.log('DEBUG: Theme =', theme, 'isDarkMode =', isDarkMode, 'legendTextColor =', legendTextColor);

    const options = {
      series: data.series,
      chart: {
        type: 'donut' as const,
        height: getChartHeight(),
        fontFamily: 'JetBrains Mono, monospace',
        background: 'transparent',
        toolbar: { 
          show: true, // Enable toolbar for zoom controls
          tools: {
            download: true,
            selection: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true
          }
        }
      },
      labels: data.labels,
      legend: {
        position: 'bottom' as const,
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '11px',
        labels: {
          colors: legendTextColor, // Use explicit color
          useSeriesColors: false
        },
        itemMargin: {
          horizontal: 8,
          vertical: 4
        }
      },
      plotOptions: {
        pie: {
          donut: {
            size: '65%'
          }
        }
      },
      colors: ['hsl(120 100% 50%)', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#10b981'],
      dataLabels: {
        enabled: true,
        style: {
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '11px',
          colors: [dataLabelColor],
          fontWeight: 'bold'
        },
        background: {
          enabled: true,
          foreColor: isDarkMode ? '#000000' : '#ffffff', // Opposite of text color for contrast
          borderColor: isDarkMode ? '#334155' : '#e2e8f0',
          borderWidth: 1,
          borderRadius: 4,
          padding: 4,
          opacity: 0.9
        },
        formatter: function(val: number) {
          return val.toFixed(1) + '%';
        }
      },
      tooltip: {
        theme: theme === 'dark' ? 'dark' : 'light'
      },
      responsive: [{
        breakpoint: 640,
        options: {
          chart: {
            height: 400
          },
          legend: {
            fontSize: '10px'
          }
        }
      }]
    };

    return (
      <div className="chart-container">
        <Chart 
          key={`topics-chart-${theme}`} 
          options={options} 
          series={data.series} 
          type="donut" 
          height={getChartHeight()} 
        />
      </div>
    );
  };

  const createModelsChart = (data: any) => {
    if (!data.models || data.models.length === 0) {
      return <p className="text-center text-muted-foreground text-sm font-mono">No model usage data found</p>;
    }

    const series = data.models.map((m: any) => m.percentage);
    const labels = data.models.map((m: any) => m.model);

    // Responsive chart height based on screen size
    const getChartHeight = () => {
      if (typeof window !== 'undefined') {
        return window.innerWidth < 640 ? 250 : window.innerWidth < 1024 ? 300 : 350;
      }
      return 300;
    };

    const options = {
      series: series,
      chart: {
        type: 'donut' as const,
        height: getChartHeight(),
        fontFamily: 'JetBrains Mono, monospace',
        background: 'transparent',
        toolbar: { 
          show: true, // Enable toolbar for zoom controls
          tools: {
            download: true,
            selection: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true
          }
        }
      },
      labels: labels,
      legend: {
        position: 'bottom' as const,
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '11px',
        labels: {
          colors: theme === 'dark' ? '#f8fafc' : '#1e293b',
          useSeriesColors: false
        },
        itemMargin: {
          horizontal: 8,
          vertical: 4
        }
      },
      plotOptions: {
        pie: {
          donut: {
            size: '65%'
          }
        }
      },
      colors: ['hsl(120 100% 50%)', '#8b5cf6', '#06b6d4', '#f59e0b', '#ef4444', '#10b981'],
      dataLabels: {
        enabled: true,
        style: {
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '11px',
          colors: [theme === 'dark' ? '#f8fafc' : '#1e293b'],
          fontWeight: 'bold'
        },
        background: {
          enabled: true,
          foreColor: theme === 'dark' ? '#1e293b' : '#f8fafc',
          borderColor: theme === 'dark' ? '#334155' : '#e2e8f0',
          borderWidth: 1,
          borderRadius: 4,
          padding: 4,
          opacity: 0.9
        }
      },
      tooltip: {
        theme: theme === 'dark' ? 'dark' : 'light'
      },
      responsive: [{
        breakpoint: 640,
        options: {
          chart: {
            height: 250
          },
          legend: {
            fontSize: '10px'
          }
        }
      }]
    };

    return (
      <div className="chart-container">
        <Chart 
          key={`models-chart-${theme}`}
          options={options} 
          series={series} 
          type="donut" 
          height={getChartHeight()} 
        />
        <div className="mt-4 max-h-60 overflow-y-auto retro-scroll">
          {data.models.map((model: any, index: number) => (
            <div key={index} className="flex justify-between items-center p-3 mb-2 bg-muted/30 retro-border rounded-md hover:bg-terminal/5 transition-colors">
              <div className="font-semibold text-sm font-mono truncate mr-2">{model.model}</div>
              <div className="text-right flex-shrink-0">
                <div className="font-bold text-terminal font-mono">{model.percentage}%</div>
                <div className="text-xs text-muted-foreground font-mono">{model.count} requests</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const createDailyChart = (data: any) => {
    if (!data.dates || data.dates.length === 0) {
      return <p className="text-center text-muted-foreground text-sm font-mono">No daily activity data found</p>;
    }

    // Debug: log the date data to understand the issue
    console.log('DEBUG: Daily chart data:', { dates: data.dates.slice(0, 5), counts: data.counts.slice(0, 5) });

    // Convert to timestamp series for a true datetime axis with robust date parsing
    const seriesData: { x: number; y: number }[] = [];
    
    data.dates.forEach((d: string, i: number) => {
      // Parse date string more robustly
      let timestamp;
      try {
        // Handle YYYY-MM-DD format specifically
        if (typeof d === 'string' && d.match(/^\d{4}-\d{2}-\d{2}$/)) {
          // Parse as UTC to avoid timezone issues
          const date = new Date(d + 'T00:00:00.000Z');
          timestamp = date.getTime();
        } else {
          // Fallback to regular Date parsing
          timestamp = new Date(d).getTime();
        }
        
        // Check if timestamp is valid
        if (isNaN(timestamp)) {
          console.warn('Invalid timestamp for date:', d);
          return;
        }
        
        seriesData.push({
          x: timestamp,
          y: data.counts[i]
        });
      } catch (error) {
        console.warn('Error parsing date:', d, error);
      }
    });

    console.log('DEBUG: Parsed series data:', seriesData.slice(0, 5));

    // Responsive chart height based on screen size
    const getChartHeight = () => {
      if (typeof window !== 'undefined') {
        return window.innerWidth < 640 ? 250 : window.innerWidth < 1024 ? 300 : 350;
      }
      return 300;
    };

    const options = {
      series: [{ name: 'Messages', data: seriesData }],
      chart: {
        type: 'area' as const,
        height: getChartHeight(),
        fontFamily: 'JetBrains Mono, monospace',
        background: 'transparent',
        toolbar: { 
          show: true, // Enable toolbar for zoom controls
          tools: {
            download: true,
            selection: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true
          }
        },
        zoom: { 
          enabled: true, 
          type: 'x' as const,
          autoScaleYaxis: true, // Auto-scale Y axis when zooming
          zoomedArea: {
            fill: {
              color: 'hsl(120 100% 50%)',
              opacity: 0.4
            },
            stroke: {
              color: 'hsl(120 100% 50%)',
              opacity: 0.4,
              width: 1
            }
          }
        },
        pan: {
          enabled: true,
          type: 'x' as const
        },
        selection: {
          enabled: true,
          type: 'x' as const
        }
      },
      xaxis: {
        type: 'datetime' as const,
        labels: {
          style: {
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '10px',
            colors: [theme === 'dark' ? '#94a3b8' : '#64748b']
          },
          datetimeUTC: false,
          // Show more granular labels based on data range
          formatter: (value: string | number, timestamp?: number) => {
            try {
              const d = new Date(typeof value === 'string' ? parseInt(value) : value);
              if (isNaN(d.getTime())) {
                return 'Invalid';
              }
              
              // For daily granularity, show month/day format
              const month = d.getMonth() + 1;
              const day = d.getDate();
              const year = d.getFullYear().toString().slice(-2);
              
              // Show different formats based on zoom level
              return `${month}/${day}/${year}`;
            } catch (error) {
              console.warn('Error formatting date label:', value, error);
              return 'Invalid';
            }
          },
          // Control label frequency
          rotate: -45, // Rotate labels to fit more
          rotateAlways: true,
          hideOverlappingLabels: false, // Show all labels, let rotation handle overlap
          showDuplicates: false
        },
        // Better tick configuration for daily data
        tickAmount: Math.min(20, Math.max(10, Math.floor(seriesData.length / 30))), // Adaptive tick count
        axisBorder: { color: theme === 'dark' ? '#334155' : '#e2e8f0' },
        axisTicks: { color: theme === 'dark' ? '#334155' : '#e2e8f0' },
        tooltip: {
          enabled: true,
          formatter: (value: string) => {
            const d = new Date(parseInt(value));
            return d.toLocaleDateString('en-US', { 
              weekday: 'short',
              year: 'numeric', 
              month: 'short', 
              day: 'numeric' 
            });
          }
        }
      },
      yaxis: {
        labels: {
          style: {
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '10px',
            colors: [theme === 'dark' ? '#94a3b8' : '#64748b']
          }
        },
        axisBorder: { color: theme === 'dark' ? '#334155' : '#e2e8f0' }
      },
      fill: {
        type: 'gradient',
        gradient: {
          shadeIntensity: 1,
          opacityFrom: 0.4,
          opacityTo: 0.1,
          stops: [0, 100],
          colorStops: [
            { offset: 0, color: 'hsl(120 100% 50%)', opacity: 0.4 },
            { offset: 100, color: 'hsl(120 100% 50%)', opacity: 0.1 }
          ]
        }
      },
      stroke: { curve: 'smooth' as const, width: 2, colors: ['hsl(120 100% 50%)'] },
      dataLabels: { enabled: false },
      grid: {
        borderColor: theme === 'dark' ? '#334155' : '#e2e8f0',
        strokeDashArray: 3,
        xaxis: { lines: { show: true } },
        yaxis: { lines: { show: true } }
      },
      tooltip: {
        theme: theme === 'dark' ? 'dark' : 'light',
        style: { fontFamily: 'JetBrains Mono, monospace' },
        x: { format: 'MMM d, yyyy' }
      },
      responsive: [{
        breakpoint: 640,
        options: { chart: { height: 250 }, xaxis: { labels: { fontSize: '9px' } } }
      }]
    };

    return (
      <div className="chart-container">
        <Chart 
          key={`daily-chart-${theme}`}
          options={options} 
          series={options.series} 
          type="area" 
          height={getChartHeight()} 
        />
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
          <div className="bg-muted/30 retro-border rounded-md p-3 text-center">
            <div className="text-terminal font-bold">{data.total_days}</div>
            <div className="text-muted-foreground">ACTIVE_DAYS</div>
          </div>
          <div className="bg-muted/30 retro-border rounded-md p-3 text-center">
            <div className="text-terminal font-bold">{data.avg_per_day}</div>
            <div className="text-muted-foreground">AVG_PER_DAY</div>
          </div>
          <div className="bg-muted/30 retro-border rounded-md p-3 text-center">
            <div className="text-terminal font-bold">{data.peak_count}</div>
            <div className="text-muted-foreground">PEAK_DAY</div>
          </div>
          <div className="bg-muted/30 retro-border rounded-md p-3 text-center">
            <div className="text-terminal font-bold">{data.total_messages}</div>
            <div className="text-muted-foreground">TOTAL_MSGS</div>
          </div>
        </div>
      </div>
    );
  };

  if (!mounted) {
    return null; // Avoid hydration mismatch
  }

  // Show loading screen when processing
  if (isLoading) {
    return <LoadingScreen progress={progress} message={statusMessage} theme={theme} />;
  }

  if (results) {
    return (
      <div className="min-h-screen bg-background">
        {/* Theme toggle - back to top right */}
        <div className="absolute top-4 right-4 z-20">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="retro-border terminal-glow font-mono"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </Button>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Header */}
          <div className="mb-8 text-center">
            <div className="flex items-center justify-center mb-4">
              <Terminal className="w-8 h-8 text-terminal mr-3" />
              <h1 className="text-2xl sm:text-3xl font-bold text-terminal font-mono">
                GPT_ANALYTICS_DASHBOARD
              </h1>
            </div>
            <p className="text-muted-foreground font-mono text-sm">
              {'{'}{'>'}{'}'}analysis_complete. displaying_insights...
            </p>
          </div>

          {/* Success message */}
          <div className="text-center mb-8">
            <p className="text-lg font-semibold text-terminal font-mono">
              INITIALIZATION_COMPLETE: Data analyzed successfully
            </p>
            {results.topics.topic_mode && (
              <p className="text-sm text-muted-foreground font-mono mt-2">
                Topic Analysis Mode: {results.topics.topic_mode === 'openai' ? 
                  'OPENAI_ENHANCED [Advanced AI Analysis]' : 
                  results.topics.topic_mode === 'bertopic' ?
                  'BERTOPIC [Advanced Semantic Analysis]' :
                  'SIMPLE [Keyword Frequency Analysis]'
                }
              </p>
            )}
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 mb-8">
            <StatCard
              icon={MessageSquare}
              title="TOTAL_MESSAGES"
              value={results.topics.message_count || 0}
              description="messages processed"
            />
            <StatCard
              icon={Database}
              title="CONVERSATIONS"
              value={results.models.conversation_count || 0}
              description="unique conversations"
            />
            <StatCard
              icon={Zap}
              title="MODELS_USED"
              value={results.models.models?.length || 0}
              description="different AI models"
            />
            <StatCard
              icon={Clock}
              title="API_REQUESTS"
              value={results.models.total_requests || 0}
              description="total requests made"
            />
          </div>

          {/* Charts Grid - improved responsive layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <ChartCard
              title="TOPIC_DISTRIBUTION"
              description={`conversation topics ${
                results.topics.topic_mode === 'openai' ? '(AI-enhanced)' : 
                results.topics.topic_mode === 'bertopic' ? '(semantic-based)' :
                '(keyword-based)'
              }`}
              icon={Terminal}
            >
              {createTopicsChart(results.topics)}
            </ChartCard>

            <ChartCard
              title="MODEL_USAGE"
              description="AI models utilized"
              icon={Zap}
            >
              {createModelsChart(results.models)}
            </ChartCard>
          </div>

          {/* Daily Activity Chart - Full Width */}
          <div className="mb-8">
            <ChartCard
              title="DAILY_MESSAGE_ACTIVITY"
              description="messages per day over time"
              icon={Clock}
            >
              {createDailyChart(results.daily)}
            </ChartCard>
          </div>

          {/* Reset button */}
          <div className="text-center">
            <Button
              variant="outline"
              onClick={() => {
                setResults(null);
                setApiKey('');
                setSelectedFile(null);
                setError(null);
              }}
              className="retro-border terminal-glow font-mono"
            >
              <Terminal className="w-4 h-4 mr-2" />
              ANALYZE_NEW_DATA
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Theme toggle - back to top right */}
      <div className="absolute top-4 right-4 z-20">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          className="retro-border terminal-glow font-mono"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </Button>
      </div>

      <div className="flex items-center justify-center min-h-screen p-4">
        <div className="w-full max-w-md sm:max-w-lg">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <Terminal className="w-8 h-8 text-terminal mr-3" />
              <h1 className="text-2xl sm:text-3xl font-bold text-terminal font-mono">
                GPT_ANALYTICS
              </h1>
            </div>
            <p className="text-muted-foreground font-mono text-sm">
              {'>'} INITIALIZE_DATA_ANALYSIS_PROTOCOL
            </p>
          </div>

          {/* Main Card */}
          <Card className="retro-border bg-card terminal-glow">
            <CardHeader className="text-center border-b border-border">
              <CardTitle className="text-xl flex items-center justify-center gap-2 font-mono">
                <Zap className="w-5 h-5 text-terminal" />
                SETUP_DASHBOARD
              </CardTitle>
              <CardDescription className="font-mono text-muted-foreground">
                configure analytics workspace
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6 p-6">
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* API Key Input */}
                <div className="space-y-2">
                  <Label htmlFor="apiKey" className="text-sm font-mono font-medium">
                    OPENAI_API_KEY (OPTIONAL):
                  </Label>
                  <Input
                    id="apiKey"
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="sk-... (leave empty for basic analysis)"
                    className="font-mono retro-border bg-muted/30"
                  />
                  <div className="space-y-2 text-sm text-muted-foreground">
                    <p><strong>With API key:</strong> AI-powered topic analysis using GPT</p>
                    <p><strong>Without API key:</strong> Advanced semantic analysis using BERTopic</p>
                  </div>
                </div>

                {/* File Upload */}
                <FileUpload onFileSelect={setSelectedFile} selectedFile={selectedFile} />

                {/* Submit Button */}
                <Button
                  type="submit"
                  disabled={isLoading}
                  className="w-full bg-terminal hover:bg-terminal/90 text-black font-mono font-bold retro-border"
                >
                  <span className="flex items-center gap-2">
                    <Terminal className="w-4 h-4" />
                    INITIALIZE_DASHBOARD
                  </span>
                </Button>
              </form>

              {/* Error Display */}
              {error && (
                <div className="bg-destructive/10 text-destructive border border-destructive/20 rounded-md p-4 font-mono text-sm">
                  {error}
                </div>
              )}

              {/* Instructions */}
              <div className="mt-6 p-4 bg-muted/30 retro-border rounded-md">
                <h3 className="font-mono font-medium text-sm mb-2 text-terminal">
                  DATA_EXPORT_INSTRUCTIONS:
                </h3>
                <ol className="text-xs text-muted-foreground space-y-1 font-mono list-decimal list-inside">
                  <li>ChatGPT → Settings → Data Controls</li>
                  <li>Click "Export data"</li>
                  <li>Wait for email with download link</li>
                  <li>Upload conversations.json file</li>
                </ol>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
} 