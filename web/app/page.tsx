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
        <Input
          id="file"
          type="file"
          accept=".json"
          onChange={handleFileChange}
          className="font-mono retro-border bg-muted/30 file:bg-terminal file:text-black file:border-0 file:rounded-md file:px-3 file:py-1 file:font-mono file:font-bold file:mr-3"
        />
        <Upload className="absolute right-3 top-3 w-4 h-4 text-muted-foreground pointer-events-none" />
      </div>
      {selectedFile && (
        <p className="text-xs text-terminal font-mono">
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
            {/* Progress Bar */}
            <div className="space-y-4">
              <div className="w-full h-3 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-terminal transition-all duration-500 relative"
                  style={{ width: `${progress}%` }}
                >
                  <div className="absolute inset-0 shimmer rounded-full" />
                </div>
              </div>
              
              <div className="text-center space-y-2">
                <div className="text-lg font-bold text-terminal font-mono">
                  {progress}%
                </div>
                <div className="text-sm text-muted-foreground font-mono">
                  {message}
                </div>
              </div>

              {/* Processing steps */}
              <div className="mt-6 space-y-2 text-xs font-mono">
                <div className={`flex items-center gap-2 ${progress > 10 ? 'text-terminal' : 'text-muted-foreground'}`}>
                  {progress > 10 ? '✓' : '○'} PARSING_JSON_DATA
                </div>
                <div className={`flex items-center gap-2 ${progress > 30 ? 'text-terminal' : 'text-muted-foreground'}`}>
                  {progress > 30 ? '✓' : '○'} EXTRACTING_CONVERSATIONS
                </div>
                <div className={`flex items-center gap-2 ${progress > 60 ? 'text-terminal' : 'text-muted-foreground'}`}>
                  {progress > 60 ? '✓' : '○'} RUNNING_TOPIC_ANALYSIS
                </div>
                <div className={`flex items-center gap-2 ${progress > 90 ? 'text-terminal' : 'text-muted-foreground'}`}>
                  {progress > 90 ? '✓' : '○'} GENERATING_INSIGHTS
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
      const response = await fetch(`http://127.0.0.1:8000/status/${jobId}`);
      const status = await response.json();

      if (status.error) {
        showError(`PROCESSING_ERROR: ${status.error}`);
        setIsLoading(false);
        return;
      }

      setProgress(status.progress || 0);
      setStatusMessage(status.ready ? 'ANALYSIS_COMPLETE' : `PROCESSING_DATA... ${status.progress || 0}%`);

      if (status.ready) {
        await loadResults(jobId);
      } else {
        setTimeout(() => pollStatus(jobId), 2000);
      }
    } catch (error) {
      showError('STATUS_CHECK_FAILED: Unable to check processing status');
      setIsLoading(false);
    }
  };

  const loadResults = async (jobId: string) => {
    try {
      const [topicsResponse, modelsResponse] = await Promise.all([
        fetch(`http://127.0.0.1:8000/topics/${jobId}`),
        fetch(`http://127.0.0.1:8000/models/${jobId}`)
      ]);

      const topicsData = await topicsResponse.json();
      const modelsData = await modelsResponse.json();

      if (topicsData.error || modelsData.error) {
        showError(`DATA_LOAD_ERROR: ${topicsData.error || modelsData.error}`);
        return;
      }

      setResults({ topics: topicsData, models: modelsData });
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

    setIsLoading(true);
    setProgress(0);
    setStatusMessage('INITIALIZING_ANALYSIS...');
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      // Only include API key if provided
      if (apiKey.trim()) {
        formData.append('api_key', apiKey);
      }

      const response = await fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (result.error) {
        showError(`UPLOAD_ERROR: ${result.error}`);
        setIsLoading(false);
        return;
      }

      setCurrentJobId(result.job_id);
      pollStatus(result.job_id);
    } catch (error) {
      showError('UPLOAD_FAILED: Unable to upload file for analysis');
      setIsLoading(false);
    }
  };

  const createTopicsChart = (data: any) => {
    if (!data.series || data.series.length === 0) {
      return <p className="text-center text-muted-foreground text-sm font-mono">No topics found in conversations</p>;
    }

    const options = {
      series: data.series,
      chart: {
        type: 'donut' as const,
        height: 300,
        fontFamily: 'JetBrains Mono, monospace',
        background: 'transparent'
      },
      labels: data.labels,
      legend: {
        position: 'bottom' as const,
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '12px',
        labels: {
          colors: [theme === 'dark' ? '#f8fafc' : '#1e293b']
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
          fontSize: '12px',
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
      }
    };

    return <Chart options={options} series={data.series} type="donut" height={300} />;
  };

  const createModelsChart = (data: any) => {
    if (!data.models || data.models.length === 0) {
      return <p className="text-center text-muted-foreground text-sm font-mono">No model usage data found</p>;
    }

    const series = data.models.map((m: any) => m.percentage);
    const labels = data.models.map((m: any) => m.model);

    const options = {
      series: series,
      chart: {
        type: 'donut' as const,
        height: 300,
        fontFamily: 'JetBrains Mono, monospace',
        background: 'transparent'
      },
      labels: labels,
      legend: {
        position: 'bottom' as const,
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: '12px',
        labels: {
          colors: [theme === 'dark' ? '#f8fafc' : '#1e293b']
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
          fontSize: '12px',
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
      }
    };

    return (
      <div>
        <Chart options={options} series={series} type="donut" height={300} />
        <div className="mt-4 max-h-60 overflow-y-auto retro-scroll">
          {data.models.map((model: any, index: number) => (
            <div key={index} className="flex justify-between items-center p-3 mb-2 bg-muted/30 retro-border rounded-md hover:bg-terminal/5 transition-colors">
              <div className="font-semibold text-sm font-mono">{model.model}</div>
              <div className="text-right">
                <div className="font-bold text-terminal font-mono">{model.percentage}%</div>
                <div className="text-xs text-muted-foreground font-mono">{model.count} requests</div>
              </div>
            </div>
          ))}
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
        {/* Theme toggle */}
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

        <div className="container-responsive py-6">
          {/* Header */}
          <div className="mb-8 text-center">
            <div className="flex items-center justify-center mb-4">
              <Terminal className="w-8 h-8 text-terminal mr-3" />
              <h1 className="text-3xl font-bold text-terminal font-mono">
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

          {/* Charts Grid */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 lg:gap-8">
            <ChartCard
              title="TOPIC_DISTRIBUTION"
              description="conversation topics analyzed"
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

          {/* Reset button */}
          <div className="mt-8 text-center">
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
      {/* Theme toggle */}
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
          <Card className="retro-border bg-card">
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
                  <p className="text-xs text-muted-foreground font-mono">
                    // optional: enables advanced topic analysis
                  </p>
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