export type McpAuthType = "none" | "bearer" | "oauth";

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
