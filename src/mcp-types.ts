export type McpAuthType = "none" | "bearer" | "cf_service_token" | "oauth";

export interface McpToolDef {
  name: string;
  description?: string;
  inputSchema?: Record<string, unknown>;
}

export interface McpServerConfig {
  id: string;
  name: string;
  endpoint: string;
  authType: McpAuthType;
  bearerToken?: string;
  cfAccessClientId?: string;
  cfAccessClientSecret?: string;
  oauthClientId?: string;
  oauthClientSecret?: string;
  oauthAuthUrl?: string;
  oauthTokenUrl?: string;
  oauthScopes?: string[];
  oauthTokens?: {
    accessToken: string;
    refreshToken?: string;
    expiresAt?: number;
  };
  enabled: boolean;
  cachedTools: McpToolDef[];
  isPreset?: boolean;
  updatedAt?: number;
}

export interface HindsightConfig {
  enabled: boolean;
  endpoint: string;
  authType: McpAuthType;
  bearerToken?: string;
  cfAccessClientId?: string;
  cfAccessClientSecret?: string;
  oauthClientId?: string;
  oauthClientSecret?: string;
  oauthTokens?: {
    accessToken: string;
    refreshToken?: string;
    expiresAt?: number;
  };
  bankId?: string; // Scope/Bank ID (default: "global" or per-conversation)
  autoRecall: boolean; // Auto pre-fetch relevant memories before turn
  autoRetain: boolean; // Auto async retain turn after response
  recallTopK?: number; // Number of memories to inject (default: 5)
  cachedTools?: McpToolDef[];
  updatedAt?: number;
}
