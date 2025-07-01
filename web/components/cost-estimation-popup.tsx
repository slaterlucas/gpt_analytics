"use client"

import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, DollarSign, Zap, X, Clock } from 'lucide-react';

interface CostBreakdown {
  total_conversations: number;
  num_clusters: number;
  use_llm_naming: boolean;
  embedding_model: string;
  costs: {
    embeddings: {
      model: string;
      tokens: number;
      cost: number;
      description: string;
      tokens_per_conversation: number;
      estimation_note: string;
    };
    llm_naming: {
      input_tokens: number;
      output_tokens: number;
      cost: number;
      description: string;
    };
    total: {
      cost: number;
      formatted: string;
    };
    high_usage_scenario: {
      embedding_tokens: number;
      embedding_cost: number;
      total_cost: number;
      cost_difference: number;
      description: string;
      tokens_per_conversation: number;
    };
  };
  cost_per_conversation: number;
  warnings: string[];
  file_info?: {
    filename: string;
    size_bytes: number;
    detected_format: string;
  };
}

interface CostEstimationPopupProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  costData: CostBreakdown | null;
  isLoading: boolean;
  error: string | null;
}

export const CostEstimationPopup: React.FC<CostEstimationPopupProps> = ({
  isOpen,
  onClose,
  onConfirm,
  costData,
  isLoading,
  error
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto retro-border bg-card terminal-glow">
        <CardHeader className="border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <DollarSign className="w-5 h-5 text-terminal" />
              <CardTitle className="font-mono text-terminal">COST_ESTIMATE</CardTitle>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="font-mono"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
          <CardDescription className="font-mono text-muted-foreground">
            estimated analysis costs with current OpenAI pricing
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-6 p-6">
          {isLoading && (
            <div className="text-center py-8">
              <div className="inline-flex items-center gap-3 text-muted-foreground font-mono">
                <div className="w-4 h-4 border-2 border-terminal border-t-transparent animate-spin rounded-full"></div>
                CALCULATING_COSTS...
              </div>
            </div>
          )}

          {error && (
            <div className="bg-destructive/10 text-destructive border border-destructive/20 rounded-md p-4">
              <div className="flex items-center gap-2 font-mono text-sm">
                <AlertTriangle className="w-4 h-4" />
                ERROR: {error}
              </div>
            </div>
          )}

          {costData && (
            <>
              {/* File Info */}
              {costData.file_info && (
                <div className="bg-muted/30 rounded-md p-4 space-y-2">
                  <h3 className="font-mono text-sm font-medium text-terminal">FILE_INFO:</h3>
                  <div className="space-y-1 text-sm font-mono text-muted-foreground">
                    <div>• Name: {costData.file_info.filename}</div>
                    <div>• Size: {(costData.file_info.size_bytes / 1024 / 1024).toFixed(1)} MB</div>
                    <div>• Format: {costData.file_info.detected_format}</div>
                    <div>• Conversations: {costData.total_conversations.toLocaleString()}</div>
                  </div>
                </div>
              )}

              {/* Analysis Settings */}
              <div className="bg-muted/30 rounded-md p-4 space-y-2">
                <h3 className="font-mono text-sm font-medium text-terminal">ANALYSIS_CONFIG:</h3>
                <div className="space-y-1 text-sm font-mono text-muted-foreground">
                  <div>• Conversations: {costData.total_conversations.toLocaleString()}</div>
                  <div>• Topic clusters: {costData.num_clusters}</div>
                  <div>• LLM naming: {costData.use_llm_naming ? 'ENABLED' : 'DISABLED'}</div>
                </div>
              </div>

              {/* Cost Breakdown */}
              <div className="space-y-4">
                <h3 className="font-mono text-sm font-medium text-terminal">COST_BREAKDOWN:</h3>
                
                <div className="space-y-3">
                  {/* Embeddings Cost */}
                  <div className="flex items-center justify-between p-3 bg-muted/20 rounded-md">
                    <div className="space-y-1">
                      <div className="font-mono text-sm text-terminal">Embeddings</div>
                      <div className="font-mono text-xs text-muted-foreground">
                        {costData.costs.embeddings.description}
                      </div>
                      <div className="font-mono text-xs text-muted-foreground">
                        ~{costData.costs.embeddings.tokens.toLocaleString()} tokens 
                        (~{costData.costs.embeddings.tokens_per_conversation} per conversation)
                      </div>
                      <div className="font-mono text-xs text-blue-600">
                        {costData.costs.embeddings.estimation_note}
                      </div>
                    </div>
                    <div className="font-mono text-lg font-bold text-terminal">
                      ${costData.costs.embeddings.cost.toFixed(4)}
                    </div>
                  </div>

                  {/* LLM Naming Cost */}
                  <div className="flex items-center justify-between p-3 bg-muted/20 rounded-md">
                    <div className="space-y-1">
                      <div className="font-mono text-sm text-terminal">
                        Topic Naming {!costData.use_llm_naming && '(Disabled)'}
                      </div>
                      <div className="font-mono text-xs text-muted-foreground">
                        {costData.costs.llm_naming.description}
                      </div>
                      {costData.use_llm_naming && (
                        <div className="font-mono text-xs text-muted-foreground">
                          ~{costData.costs.llm_naming.input_tokens} in + {costData.costs.llm_naming.output_tokens} out tokens
                        </div>
                      )}
                    </div>
                    <div className="font-mono text-lg font-bold text-terminal">
                      ${costData.costs.llm_naming.cost.toFixed(4)}
                    </div>
                  </div>

                  {/* High Usage Scenario */}
                  {costData.costs.high_usage_scenario.cost_difference > 0.01 && (
                    <div className="flex items-center justify-between p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-md">
                      <div className="space-y-1">
                        <div className="font-mono text-sm text-yellow-700">High-Usage Scenario</div>
                        <div className="font-mono text-xs text-yellow-600">
                          {costData.costs.high_usage_scenario.description}
                        </div>
                        <div className="font-mono text-xs text-yellow-600">
                          ~{costData.costs.high_usage_scenario.embedding_tokens.toLocaleString()} tokens 
                          (~{costData.costs.high_usage_scenario.tokens_per_conversation} per conversation)
                        </div>
                      </div>
                      <div className="font-mono text-lg font-bold text-yellow-700">
                        ${costData.costs.high_usage_scenario.total_cost.toFixed(4)}
                      </div>
                    </div>
                  )}

                  {/* Total Cost */}
                  <div className="flex items-center justify-between p-4 bg-terminal/10 border border-terminal/20 rounded-md">
                    <div className="space-y-1">
                      <div className="font-mono text-base font-bold text-terminal">TOTAL_COST</div>
                      <div className="font-mono text-xs text-muted-foreground">
                        ${(costData.cost_per_conversation * 1000).toFixed(4)} per 1K conversations
                      </div>
                      {costData.costs.high_usage_scenario.cost_difference > 0.01 && (
                        <div className="font-mono text-xs text-yellow-600">
                          Range: {costData.costs.total.formatted} - ${costData.costs.high_usage_scenario.total_cost.toFixed(3)}
                        </div>
                      )}
                    </div>
                    <div className="font-mono text-2xl font-bold text-terminal">
                      {costData.costs.total.formatted}
                    </div>
                  </div>
                </div>
              </div>

              {/* Warnings */}
              {costData.warnings.length > 0 && (
                <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-md p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-500" />
                    <span className="font-mono text-sm font-medium text-yellow-500">WARNINGS:</span>
                  </div>
                  <div className="space-y-1">
                    {costData.warnings.map((warning, index) => (
                      <div key={index} className="font-mono text-sm text-yellow-600">
                        • {warning}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Estimated Processing Time */}
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-md p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Clock className="w-4 h-4 text-blue-500" />
                  <span className="font-mono text-sm font-medium text-blue-500">ESTIMATED_TIME:</span>
                </div>
                <div className="font-mono text-sm text-blue-600">
                  ~{Math.max(1, Math.ceil(costData.total_conversations / 1000))} minute{Math.ceil(costData.total_conversations / 1000) !== 1 ? 's' : ''} for {costData.total_conversations.toLocaleString()} conversations
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-4">
                <Button
                  variant="outline"
                  onClick={onClose}
                  className="flex-1 font-mono retro-border"
                >
                  CANCEL
                </Button>
                <Button
                  onClick={onConfirm}
                  className="flex-1 bg-terminal hover:bg-terminal/90 text-black font-mono font-bold retro-border"
                >
                  <Zap className="w-4 h-4 mr-2" />
                  CONFIRM & ANALYZE
                </Button>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}; 