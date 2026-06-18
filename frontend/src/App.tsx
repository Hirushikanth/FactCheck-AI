import { useState, createContext, useContext } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  IconShieldCheck,
  IconActivity,
  IconAlertCircle,
} from "@tabler/icons-react";
import { SessionScreen } from "./screens/SessionScreen";
import { ResultsScreen } from "./screens/ResultsScreen";
import { HistoryScreen } from "./screens/HistoryScreen";
import { useHealth } from "./hooks/useHealth";
import type { SessionDetail } from "./api/types";

// ── Global app context ────────────────────────────────────────────────────────
export type AppTab = "session" | "results" | "history";

interface AppContextValue {
  activeTab: AppTab;
  setActiveTab: (tab: AppTab) => void;
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
  activeSession: SessionDetail | null;
  setActiveSession: (s: SessionDetail | null) => void;
}

const AppContext = createContext<AppContextValue>({} as AppContextValue);
export const useApp = () => useContext(AppContext);

// ── Query client ──────────────────────────────────────────────────────────────
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      retry: 1,
    },
  },
});

// ── Root app wrapped with providers ──────────────────────────────────────────
export default function AppRoot() {
  return (
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState<AppTab>("session");
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [activeSession, setActiveSession] = useState<SessionDetail | null>(null);

  const ctx: AppContextValue = {
    activeTab,
    setActiveTab,
    activeSessionId,
    setActiveSessionId,
    activeSession,
    setActiveSession,
  };

  function switchTab(tab: AppTab) {
    setActiveTab(tab);
  }

  return (
    <AppContext.Provider value={ctx}>
      <div className="app-shell">
        <TopBar activeTab={activeTab} onTabChange={switchTab} />
        <div className="app-body">
          {activeTab === "session" && <SessionScreen />}
          {activeTab === "results" && <ResultsScreen />}
          {activeTab === "history" && <HistoryScreen />}
        </div>
      </div>
    </AppContext.Provider>
  );
}

// ── Top Bar ───────────────────────────────────────────────────────────────────
function TopBar({
  activeTab,
  onTabChange,
}: {
  activeTab: AppTab;
  onTabChange: (t: AppTab) => void;
}) {
  const { data: health, isError } = useHealth();

  const healthOk = health?.status === "ok";
  const healthChecking = !health && !isError;

  return (
    <header className="topbar">
      {/* Logo */}
      <div className="topbar-logo">
        <div className="logo-mark">
          <IconShieldCheck size={15} stroke={2} />
        </div>
        <span className="logo-text">
          FactCheck <span className="logo-text-dim">AI</span>
        </span>
      </div>

      {/* Nav tabs */}
      <nav className="nav-tabs" aria-label="Main navigation">
        {(["session", "results", "history"] as AppTab[]).map((tab) => (
          <button
            key={tab}
            className={`nav-tab${activeTab === tab ? " active" : ""}`}
            onClick={() => onTabChange(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>

      {/* Right: health + avatar */}
      <div className="topbar-right">
        <div
          className="health-indicator"
          title={
            healthChecking
              ? "Checking backend…"
              : healthOk
                ? `Backend ok · ${health.ollama_model}`
                : "Backend unreachable"
          }
        >
          {healthChecking ? (
            <IconActivity size={14} className="health-icon checking" />
          ) : healthOk ? (
            <IconActivity size={14} className="health-icon ok" />
          ) : (
            <IconAlertCircle size={14} className="health-icon error" />
          )}
        </div>
        <div className="avatar" aria-label="User avatar">
          H
        </div>
      </div>
    </header>
  );
}
