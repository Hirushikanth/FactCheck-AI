import { createContext, useContext } from "react";
import type { SessionDetail } from "./api/types";

export type AppTab = "session" | "results" | "history";

export interface AppContextValue {
  activeTab: AppTab;
  setActiveTab: (tab: AppTab) => void;
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
  activeSession: SessionDetail | null;
  setActiveSession: (s: SessionDetail | null) => void;
}

export const AppContext = createContext<AppContextValue>({} as AppContextValue);
export const useApp = () => useContext(AppContext);
