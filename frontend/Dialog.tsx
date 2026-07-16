

import * as React from "react";
import "./../assets/scss/App.scss";
import { ActionButton, DefaultButton, IconButton, Modal, Stack } from "@fluentui/react";
import { observer } from "mobx-react";
import { getPaperByTitle } from "./../request";
import Markdown from "react-markdown";
import { Logger } from "../socket/logger";
import { API_BASE_URL } from '../config';

const baseUrl = `${API_BASE_URL}/`;

// Ensure we only reset chat state and backend memory once per page load,
// not every time a new chat tab (Dialog instance) mounts.
let hasInitializedChatMemory = false;

// Define getPaperById at the top level
export async function getPaperById(id: string) {
  const res = await fetch(`${baseUrl}getPaperById?id=${id}`);
  return res.json();
}

const parseTitleAndId = (raw: string) => {
  // 1) Extract ID (various formats supported)
  const idMatch = raw.match(/(?:<!--ID:([\w-]+)-->|\[\[ID:([^\]]+)\]\])/i);
  const extractedId = idMatch ? (idMatch[1] || idMatch[2]) : null;

  // 2) Extract title
  const titleMatch = raw.match(/Title:\s*([^\n\.<]+)/i);
  const title = titleMatch ? titleMatch[1].trim() : null;

  // 3) Build fallback key (only for UI, never used in backend queries)
  const fallbackKey = extractedId || (title
    ? `temp_${title.replace(/\s+/g, "_").toLowerCase()}`
    : `temp_raw_${raw.slice(0, 20).replace(/\s+/g, "_")}`);

  return { id: extractedId, title, key: fallbackKey };
};

// Build a map of paperId → {title, source} from paper-block lines in a message
const buildRefMap = (text: string): Map<string, { title: string; source: string }> => {
  const map = new Map<string, { title: string; source: string }>();
  const lines = text.split("\n");
  let lastId: string | null = null;
  for (const line of lines) {
    const m = line.match(/Title:\s*([^\[]+)\s*\[\[ID:([^\]]+)\]\]/i);
    if (m) {
      lastId = m[2].trim();
      map.set(lastId, { title: m[1].replace(/\*\*/g, "").trim(), source: "" });
    } else if (lastId) {
      const src = line.match(/[-\s*]*Source:\s*(.+)/i);
      if (src) {
        const entry = map.get(lastId);
        if (entry) map.set(lastId, { ...entry, source: src[1].trim() });
        lastId = null;
      }
    }
  }
  return map;
};

const CitationBadge = ({ title: initialTitle, source: initialSource, paperId }: { title: string; source: string; paperId: string }) => {
  const [hovered, setHovered] = React.useState(false);
  const [fetchedTitle, setFetchedTitle] = React.useState("");
  const [fetchedSource, setFetchedSource] = React.useState("");
  const fetchStarted = React.useRef(false);

  const isArxiv      = paperId.startsWith("arxiv_");
  const isOpenalex   = /^W\d+$/.test(paperId);
  const isOpenreview = paperId.startsWith("or_");

  // title/source are the reactive values used for display and URL building.
  // fetchedTitle is set lazily on hover when initialTitle is absent.
  const title  = fetchedTitle  || initialTitle;
  const source = fetchedSource || initialSource;

  // url must be derived from the reactive `title` (not the prop `initialTitle`) so the
  // Google Scholar fallback URL updates after fetchedTitle is set on hover.
  const hasSourceUrl = isArxiv || isOpenalex || isOpenreview;
  const url = isArxiv
    ? `https://arxiv.org/abs/${paperId.replace("arxiv_", "")}`
    : isOpenalex
    ? `https://openalex.org/${paperId}`
    : isOpenreview
    ? `https://openreview.net/forum?id=${paperId.replace("or_", "")}`
    : title
    ? `https://scholar.google.com/scholar?q=${encodeURIComponent(title)}`
    : "";

  const displaySource = source || (isArxiv ? "arXiv" : isOpenalex ? "OpenAlex" : isOpenreview ? "OpenReview" : "Source");
  const shortSource   = displaySource.length > 14 ? displaySource.slice(0, 12) + "…" : displaySource;
  const letter        = displaySource[0]?.toUpperCase() || "S";

  const handleMouseEnter = React.useCallback(() => {
    setHovered(true);
    // Skip fetch only when:
    //   - already fetched, OR
    //   - the ID maps to a known source URL (no title needed for the URL) AND we already have a title
    if (fetchStarted.current) return;
    if (hasSourceUrl && initialTitle) return;
    fetchStarted.current = true;
    getPaperById(paperId)
      .then((paper: any) => {
        if (paper?.Title)  setFetchedTitle(paper.Title);
        if (paper?.Source) setFetchedSource(paper.Source);
      })
      .catch(() => {});
  }, [paperId, initialTitle, hasSourceUrl]);

  return (
    <span
      style={{ position: "relative", display: "inline-block", verticalAlign: "middle", margin: "0 2px" }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setHovered(false)}
    >
      <a
        href={url || undefined}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          display: "inline-flex", alignItems: "center", gap: 3,
          padding: "1px 8px 1px 6px", borderRadius: 20,
          background: hovered ? "#e4e4e4" : "#f0f0f0",
          color: "#333", fontSize: "0.78rem", textDecoration: "none",
          border: "1px solid #e0e0e0", lineHeight: 1.6, whiteSpace: "nowrap",
          transition: "background 0.15s",
        }}
      >
        {shortSource}
        {url && <span style={{ fontSize: "0.65rem", color: "#888", marginLeft: 1 }}>↗</span>}
      </a>
      {hovered && (
        <div style={{
          position: "absolute", bottom: "calc(100% + 6px)", left: "50%",
          transform: "translateX(-50%)",
          background: "#fff", border: "1px solid #e5e5e5", borderRadius: 12,
          boxShadow: "0 4px 20px rgba(0,0,0,0.13)", padding: "12px 14px",
          minWidth: 240, maxWidth: 320, zIndex: 9999, pointerEvents: "none", textAlign: "left",
        }}>
          <div style={{ fontSize: "0.87rem", color: "#1a1a1a", lineHeight: 1.45, marginBottom: 8 }}>
            {title || paperId}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{
              width: 18, height: 18, borderRadius: "50%", background: "#1a1a1a", color: "#fff",
              fontSize: "0.6rem", fontWeight: 700,
              display: "inline-flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
            }}>
              {letter}
            </span>
            <span style={{ fontSize: "0.8rem", color: "#666" }}>{displaySource}</span>
          </div>
        </div>
      )}
    </span>
  );
};

// Matches [[ID:xxx]] (backend format) AND [[xxx]] (LLM inline citation format)
const CITE_REGEX     = /(\[\[(?:ID:)?[^\]]+\]\])/g;
const CITE_EXTRACT   = /\[\[(?:ID:)?([^\]]+)\]\]/;
const CITE_STRIP     = /\[\[(?:ID:)?[^\]]+\]\]/g;

// Extract a clean, linkable ID from whatever the LLM stuffed inside [[...]].
// The LLM sometimes appends the paper title: "W4401857375: A Survey on RAG" or
// "W4389984066; Corrective RAG" — strip everything after the real ID.
const normalizeId = (raw: string): string => {
  const t = raw.trim();
  const extracted =
    t.match(/^(W\d+)/)?.[1]                  // OpenAlex  W4401857375…
    ?? t.match(/^(arxiv_[^\s;:,]+)/)?.[1]    // ArXiv     arxiv_2301.00001v2…
    ?? t.match(/^(or_[^\s;:,]+)/)?.[1]       // OpenReview or_abc123…
    ?? t.split(/[;:,\s]/)[0].trim();         // Unknown — take first token
  return extracted || t;
};

// Expand grouped citations before Markdown parses the text.
// The LLM sometimes groups multiple IDs inside one block:
//   [[W1234]; [W5678]; [W9999]]  →  [[W1234]] [[W5678]] [[W9999]]
// Without this, CITE_REGEX (which uses [^\]]+) stops at the first ] and
// the whole grouped block is left as raw visible text.
const expandGroupedCitations = (text: string): string =>
  text.replace(
    /\[\[([^\[\]]*(?:\];\s*\[[^\[\]]*)*)\]\]/g,
    (_: string, content: string) => {
      const ids = content.split(/\];\s*\[/).map((s: string) => s.trim());
      return ids.map((id: string) => `[[${id}]]`).join(" ");
    }
  );

const renderInlineCitations = (
  raw: string,
  refMap: Map<string, { title: string; source: string }>
): React.ReactNode[] =>
  raw.split(CITE_REGEX).map((part, i) => {
    const m = part.match(CITE_EXTRACT);
    if (m) {
      const id = normalizeId(m[1]);
      const ref = refMap.get(id);
      return <React.Fragment key={i}><CitationBadge title={ref?.title || ""} source={ref?.source || ""} paperId={id} /></React.Fragment>;
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });

////////////////////////////////////////////////////////



export const Dialog = observer(({ props }) => {
  const {
    chatText,
    chatHistory,
    chatSelectedPaper,
    displayMessages: savedDisplayMessages,
    updateDialogState,
    addToSelectNodeIDs,
    addToSimilarInputPapers,
    addToSavedPapers,
    isInSimilarInputPapers,
    isInSavedPapers,
    isInSelectedNodeIDs,
    tabId, // <-- NEW: needed for DATA_SIGNAL sidebar feature
    /** From corpus Results list: add paper to "Ask" selection for next message */
    queuedCorpusQuestionPaper,
    onConsumeQueuedCorpusQuestionPaper,
  } = props;

  // --- SIDEBAR STATE (NEW) ---
  const [sidebarPapers, setSidebarPapers] = React.useState<any[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = React.useState(false);
  const [papersToShow, setPapersToShow] = React.useState(20);

  const [displayMessages, setDisplayMessages] = React.useState<any[]>(
    chatHistory || []
  );

  const [retrievedPapers, setRetrievedPapers] = React.useState<{ title: string, id: string | null, key: string }[]>([]);

  // isWaiting: true while the "thinking" sentinel is present in chatHistory.
  // agentStatus: local state so STATUS signals update the label directly without
  // going through the MobX→prop→useEffect chain (which batches away intermediate renders).
  const thinkingEntry = displayMessages.find((m: any) => m.role === "thinking");
  const isWaiting: boolean = !!thinkingEntry;

  // "Something else" / "Rephrase" custom inputs inside clarification cards (ephemeral).
  const [customExpansionText, setCustomExpansionText] = React.useState<string | null>(null);
  const [customConfirmText, setCustomConfirmText] = React.useState<string | null>(null);
  const [agentStatus, setAgentStatus] = React.useState<string>("");
  React.useEffect(() => { console.log("[agentStatus render]", agentStatus); }, [agentStatus]);

  // When the tab switches (tabId changes) clear ephemeral card input state.
  React.useEffect(() => {
    setCustomExpansionText(null);
    setCustomConfirmText(null);
    setAgentStatus("");
  }, [tabId]);

  // On first mount (including browser refresh / first visit), clear all chat
  // state on the frontend *and* reset backend memory so every load starts clean.
  // IMPORTANT: Guard with a module-level flag so opening additional chat tabs
  // does NOT wipe existing chats.
  React.useEffect(() => {
    if (hasInitializedChatMemory) {
      return;
    }
    hasInitializedChatMemory = true;

    // Clear local UI state
    setDisplayMessages([]);
    setRetrievedPapers([]);

    // Clear dialog store state (also drops any thinking sentinel in chatHistory)
    updateDialogState({
      chatHistory: [],
      chatText: "",
    });

    // Clear server-side chat memory for this app instance
    fetch(`${baseUrl}resetMemory`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    }).catch((err) => {
      console.error("Failed to reset memory on mount:", err);
    });
  }, []);

  // Paper info modal state
  const [isModalOpen, setModalState] = React.useState(false);
  const [paperInfo, setPaperInfo] = React.useState<any>(null);
  const [loadingPaperInfo, setLoadingPaperInfo] = React.useState(false);

  // Function to open paper info modal
  const openPaperInfoModal = async (title: string, id: string | null) => {
    setLoadingPaperInfo(true);
    setModalState(true);

    try {
      let paper = null;
      if (id) {
        paper = await getPaperById(id);
      } else {
        const papers = await getPaperByTitle(title);
        paper = papers.length > 0 ? papers[0] : null;
      }

      if (paper) {
        setPaperInfo(paper);
        Logger.logUIInteraction({
          component: "Dialog",
          action: "paperInfoModalOpen",
          paperTitle: title,
          paperId: paper.ID || id,
        });
      } else {
        setPaperInfo(null);
      }
    } catch (error) {
      console.error("Error fetching paper info:", error);
      setPaperInfo(null);
    }
    setLoadingPaperInfo(false);
  };

  React.useEffect(() => {
    setDisplayMessages(chatHistory || []);
  }, [chatHistory]);

  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  /** Local draft so typing does not call updateDialogState → App re-render on every key */
  const [draftChatText, setDraftChatText] = React.useState(() => chatText ?? "");
  React.useEffect(() => {
    setDraftChatText(chatText ?? "");
  }, [chatText]);

  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  const prevMessageCountRef = React.useRef(0);

  // When switching chat window, scroll to latest content
  React.useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [tabId]);

  // Scroll to bottom only when the user sends a new message (not while model is generating)
  React.useEffect(() => {
    const prevLen = prevMessageCountRef.current;
    prevMessageCountRef.current = displayMessages.length;
    const addedMessage = displayMessages.length > prevLen;
    const lastIsUser = displayMessages.length > 0 && displayMessages[displayMessages.length - 1].role === "user";
    if (addedMessage && lastIsUser && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }

    // Reset button states when a new AI message is added
    if (displayMessages.length > 0 && displayMessages[displayMessages.length - 1].role === "ai") {
      setButtonsClicked({ locateAll: false, addAll: false, saveAll: false });
    }
  }, [displayMessages]);

  // Handle Enter key to send message
  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (draftChatText.trim()) {
        Logger.logUIInteraction({
          component: "Dialog",
          action: "askButtonClick",
          value: draftChatText,
          chatHistoryLength: chatHistory.length,
          trigger: "enter_key",
        });
        chatRequest();
      }
    }
  };

  const onChangeChatText = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newText = e.target.value;
    setDraftChatText(newText);

    // Auto-grow the textarea up to ~3 lines, then scroll
    if (textareaRef.current) {
      const el = textareaRef.current;
      el.style.height = "auto";
      const lineHeight = 24; // keep in sync with CSS line-height
      const maxHeight = lineHeight * 3;
      const newHeight = Math.min(el.scrollHeight, maxHeight);
      el.style.height = `${newHeight}px`;
      el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
    }
  };

  const currentAIResponseText = React.useCallback(() => {
    for (let i = displayMessages.length - 1; i >= 0; i--) {
      if (displayMessages[i].role === "ai") return displayMessages[i].text || "";
    }
    return "";
  }, [displayMessages]);

  // Extract all papers from the current AI response
  const extractAllPapers = React.useCallback(() => {
    const text = currentAIResponseText();
    const papers: { title: string; id: string | null }[] = [];

    const titlePattern = /\*\*Title:\*\*\s*([^\[]+)\s*\[\[ID:([^\]]+)\]\]/gi;
    let match;

    while ((match = titlePattern.exec(text)) !== null) {
      const title = match[1].trim();
      const id = match[2].trim();
      papers.push({ title, id });
    }

    const listPattern = /Title:\s*([^\[]+)\s*\[\[ID:([^\]]+)\]\]/gi;
    while ((match = listPattern.exec(text)) !== null) {
      const title = match[1].trim();
      const id = match[2].trim();
      if (!papers.some(p => p.id === id)) {
        papers.push({ title, id });
      }
    }

    return papers;
  }, [currentAIResponseText]);

  // State to track which buttons have been clicked
  const [buttonsClicked, setButtonsClicked] = React.useState({
    locateAll: false,
    addAll: false,
    saveAll: false
  });

  // Papers the user wants to reference explicitly in their next question
  const [selectedQuestionPapers, setSelectedQuestionPapers] = React.useState<
    { id: string | null; title: string }[]
  >([]);

  const toggleQuestionPaper = React.useCallback(
    (paper: { id: string | null; title: string }) => {
      setSelectedQuestionPapers((prev) => {
        const exists = prev.some(
          (p) => (p.id && p.id === paper.id) || (!p.id && p.title === paper.title)
        );
        if (exists) {
          return prev.filter(
            (p) => !((p.id && p.id === paper.id) || (!p.id && p.title === paper.title))
          );
        }
        return [...prev, paper];
      });
    },
    []
  );

  // Results list "Ask" queues a paper; ref avoids duplicate toggles under React Strict Mode
  const consumedCorpusAskTokenRef = React.useRef<number | null>(null);
  React.useEffect(() => {
    const q = queuedCorpusQuestionPaper;
    if (!q) {
      return;
    }
    if (consumedCorpusAskTokenRef.current === q.token) {
      return;
    }
    consumedCorpusAskTokenRef.current = q.token;
    toggleQuestionPaper({
      id: String(q.id),
      title: q.title,
    });
    onConsumeQueuedCorpusQuestionPaper?.();
    // Only re-run when a new queue token is set
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queuedCorpusQuestionPaper?.token, toggleQuestionPaper]);

  const clearSelectedQuestionPapers = React.useCallback(() => {
    setSelectedQuestionPapers([]);
  }, []);

  const buildQuestionPrompt = React.useCallback(
    (questionText: string) => {
      if (!selectedQuestionPapers.length) {
        return questionText;
      }
      const headerLines = selectedQuestionPapers.map((p, idx) =>
        `${idx + 1}. ${p.title}${p.id ? ` (ID: ${p.id})` : ""}`
      );
      const header = `Selected papers:\n${headerLines.join("\n")}`;
      return `${header}\n\nQuestion: ${questionText}`;
    },
    [selectedQuestionPapers]
  );


  // Core chat sender used by both the text box and inline "Load more" link.
  const sendChat = (text: string) => {
    if (!text.trim()) return;

    // Clear any pending clarification cards and transient input state
    setCustomExpansionText(null);
    setCustomConfirmText(null);

    // Combine any selected papers with the user's question so the model
    // sees both the paper titles and the free-form question.
    const finalText = buildQuestionPrompt(text);

    // Draft is local-only while typing; parent chatText often stays "" so the
    // sync effect does not run — clear the textarea immediately on send.
    setDraftChatText("");

    // 1. Clear previous retrieval state
    setRetrievedPapers([]);

    // 2. Optimistically add the USER message and the thinking sentinel to parent
    // chatHistory so both survive split-pane remounts.
    const userMessage = {
      role: "user" as const,
      text,
      referencedPapers: selectedQuestionPapers.length ? [...selectedQuestionPapers] : undefined,
    };

    updateDialogState((prev: any) => ({
      // Drop any signal cards and old thinking sentinel; add user msg + new thinking sentinel
      chatHistory: [
        ...(prev.chatHistory || []).filter((m: any) => m.role !== "signal" && m.role !== "thinking"),
        userMessage,
        { role: "thinking" },
      ],
      chatText: "", // clear the input box
    }));

    // Reset input height back to one row and clear selected papers after sending
    if (textareaRef.current) {
      textareaRef.current.style.height = "";
      textareaRef.current.style.overflowY = "hidden";
    }
    clearSelectedQuestionPapers();

    const baseHistory = [...chatHistory.filter((m: any) => m.role !== "signal"), userMessage];

    // 3. Start the Fetch Request
    fetch(`${baseUrl}chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: tabId,
        text: finalText,
        chat_history_raw: baseHistory,
      }),
    }).then((response) => {
      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let partialResponse = "";
      let isFirstChunk = true;

      const readChunk = ({ done, value }: ReadableStreamReadResult<Uint8Array>) => {
        const SIGNAL_SHOW_LOAD_MORE = "[SIGNAL:SHOW_LOAD_MORE]";
        const signalRegex = new RegExp(
          "\\s*" + SIGNAL_SHOW_LOAD_MORE.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\s*",
          "g"
        );

        // --- CASE A: Stream is Finished ---
        if (done) {
          // Append any final chunk (signal or text often arrives with done=true)
          if (value && value.length > 0) {
            partialResponse += decoder.decode(value);
          }
          // Remove the thinking sentinel and clear the status label.
          setAgentStatus("");
          updateDialogState((prev: any) => ({
            chatHistory: (prev.chatHistory || []).filter((m: any) => m.role !== "thinking"),
          }));

          // Persist "load more" when backend sent signal OR when response looks like a paper list (fallback)
          const HAS_MORE_MARKER = "[[HAS_MORE_PAPERS]]";
          const hadSignal = partialResponse.includes(SIGNAL_SHOW_LOAD_MORE);
          const looksLikePaperList = /Title:\s*/i.test(partialResponse) && /\[\[ID:[^\]]+\]\]/.test(partialResponse);
          if (hadSignal || looksLikePaperList) {
            const cleaned = hadSignal ? partialResponse.replace(signalRegex, "").trim() : partialResponse.trim();
            const textWithMarker = cleaned + "\n" + HAS_MORE_MARKER;
            if (hadSignal) partialResponse = cleaned;
            updateDialogState((prev: any) => {
              const history = [...(prev.chatHistory || [])];
              const lastIdx = history.length - 1;
              if (lastIdx >= 0 && history[lastIdx].role === "ai") {
                history[lastIdx] = { ...history[lastIdx], text: textWithMarker, hasMorePapers: true };
              }
              return { chatHistory: history };
            });
          }

          const papers: { title: string; id: string | null; key: string }[] = [];
          const lines = partialResponse.split("\n");

          for (const line of lines) {
            const { id, title, key } = parseTitleAndId(line);
            if (key && !papers.some((p) => p.key === key)) {
              if (title) papers.push({ id, title, key });
            }
          }
          console.log("Stream complete. Papers found:", papers);
          setRetrievedPapers(papers);
          return;
        }

        // --- CASE B: Process New Chunk ---
        let newTextChunk = decoder.decode(value);
        console.log("[STREAM chunk]", JSON.stringify(newTextChunk.slice(0, 120)));

        // --- INTERCEPT SIGNAL TAGS (e.g. [SIGNAL:SHOW_LOAD_MORE]) ---
        if (newTextChunk.includes(SIGNAL_SHOW_LOAD_MORE)) {
          newTextChunk = newTextChunk.replace(signalRegex, "");
        }

        // --- INTERCEPT STATUS SIGNALS [SIGNAL:STATUS:...] ---
        if (newTextChunk.includes("[SIGNAL:STATUS:")) {
          const sm = newTextChunk.match(/\[SIGNAL:STATUS:([^\]]+)\]/);
          if (sm) {
            console.log("[STATUS] setting agentStatus →", sm[1]);
            setAgentStatus(sm[1]);
          }
          newTextChunk = newTextChunk.replace(/\[SIGNAL:STATUS:[^\]]+\]/g, "");
        }

        // --- INTERCEPT QUERY EXPANSION SIGNAL ---
        const EXPANSION_PREFIX = "[SIGNAL:QUERY_EXPANSION:";
        if (newTextChunk.includes(EXPANSION_PREFIX)) {
          const startIdx = newTextChunk.indexOf(EXPANSION_PREFIX);
          const jsonStart = startIdx + EXPANSION_PREFIX.length;
          let depth = 0;
          let jsonEnd = -1;
          for (let i = jsonStart; i < newTextChunk.length; i++) {
            if (newTextChunk[i] === "{") depth++;
            else if (newTextChunk[i] === "}") { depth--; if (depth === 0) { jsonEnd = i + 1; break; } }
          }
          if (jsonEnd > 0) {
            try {
              const data = JSON.parse(newTextChunk.slice(jsonStart, jsonEnd));
              if (data.expansions?.length) {
                // Store signal card and remove thinking sentinel in one update
                updateDialogState((prev: any) => ({
                  chatHistory: [
                    ...(prev.chatHistory || []).filter((m: any) => m.role !== "thinking"),
                    { role: "signal", signalType: "query_expansion", signalData: data },
                  ],
                }));
              }
            } catch (_) { /* malformed JSON — ignore */ }
            const tagEnd = newTextChunk.indexOf("]", jsonEnd) + 1;
            newTextChunk = newTextChunk.slice(0, startIdx) + (tagEnd > 0 ? newTextChunk.slice(tagEnd) : "");
          }
        }

        // --- INTERCEPT INTENT CONFIRM SIGNAL ---
        const CONFIRM_PREFIX = "[SIGNAL:INTENT_CONFIRM:";
        if (newTextChunk.includes(CONFIRM_PREFIX)) {
          const startIdx = newTextChunk.indexOf(CONFIRM_PREFIX);
          const jsonStart = startIdx + CONFIRM_PREFIX.length;
          let depth = 0, jsonEnd = -1;
          for (let i = jsonStart; i < newTextChunk.length; i++) {
            if (newTextChunk[i] === "{") depth++;
            else if (newTextChunk[i] === "}") { depth--; if (depth === 0) { jsonEnd = i + 1; break; } }
          }
          if (jsonEnd > 0) {
            try {
              const data = JSON.parse(newTextChunk.slice(jsonStart, jsonEnd));
              if (data.items?.length) {
                // Store signal card and remove thinking sentinel in one update
                updateDialogState((prev: any) => ({
                  chatHistory: [
                    ...(prev.chatHistory || []).filter((m: any) => m.role !== "thinking"),
                    { role: "signal", signalType: "intent_confirm", signalData: data },
                  ],
                }));
              }
            } catch (_) { /* malformed JSON — ignore */ }
            const tagEnd = newTextChunk.indexOf("]", jsonEnd) + 1;
            newTextChunk = newTextChunk.slice(0, startIdx) + (tagEnd > 0 ? newTextChunk.slice(tagEnd) : "");
          }
        }

        // --- REGULAR TEXT STREAM (after signals stripped) ---
        partialResponse += newTextChunk;

        if (isFirstChunk) {
          if (newTextChunk.trim()) {
            // First real text chunk: remove thinking sentinel and create AI bubble in ONE
            // updateDialogState call so the parent always sees them together.
            isFirstChunk = false;
            updateDialogState((prev: any) => ({
              chatHistory: [
                ...(prev.chatHistory || []).filter((m: any) => m.role !== "thinking"),
                { role: "ai", text: newTextChunk },
              ],
            }));
            reader.read().then(readChunk);
            return;
          }
        }

        if (newTextChunk.trim()) {
          updateDialogState((prev: any) => {
            const historyCopy = [...prev.chatHistory];
            const lastMsgIndex = historyCopy.length - 1;
            const lastMsg = historyCopy[lastMsgIndex];

            if (lastMsg && lastMsg.role === "ai") {
              let newText = lastMsg.text + newTextChunk;
              newText = newText.replace(signalRegex, ""); // strip signal (handles split across chunks)
              historyCopy[lastMsgIndex] = {
                ...lastMsg,
                text: newText,
              };
            }
            return { chatHistory: historyCopy };
          });
        }

        reader.read().then(readChunk);
      };

      reader.read().then(readChunk);
    }).catch(err => {
      console.error("Stream error:", err);
      setAgentStatus("");
      updateDialogState((prev: any) => ({
        chatHistory: (prev.chatHistory || []).filter((m: any) => m.role !== "thinking"),
      }));
    });
  };

  // --- Public handler for the text box / Ask button ---
  const chatRequest = () => {
    if (!draftChatText.trim()) return;
    sendChat(draftChatText);
  };



  const extractText = (children: React.ReactNode): string => {
    if (typeof children === "string") return children;
    if (Array.isArray(children)) {
      const joined = children.map((child: React.ReactNode) => extractText(child)).join(" ");
      // Markdown parses [[W1234]] as "[" + "[W1234]" (ref link) + "]", joined with spaces
      // as "[ [W1234] ]". Repair that back to "[[W1234]]" so CITE_REGEX can match.
      return joined.replace(/\[\s*\[([^\]]*)\]\s*\]/g, "[[$1]]");
    }
    if (React.isValidElement(children)) {
      return extractText(children.props.children);
    }
    return "";
  };

  const renderPaperBlock = (title: string, id: string | null, raw: string) => {
    // 1. If it's not a valid paper title line, just render cleaned text
    if (!title || title === raw || !/Title:\s*[^\n]+/i.test(raw)) {
      const cleaned = raw.replace(/\[\[ID:[^\]]+\]\]/g, "");
      return <span>{cleaned}</span>;
    }

    // 2. Clean the title (remove ID tags)
    const cleanTitle = title.replace(/\[\[ID:[^\]]+\]\]/g, "").trim();

    // 3. Extract the Summary
    const summary = raw
      .replace(/\*\*Title:[\s\S]*?\[\[ID:[^\]]+\]\]\*\*/gi, "")
      .replace(/Title:\s*[^\n]+/gi, "")
      .replace(/<!--ID:[^>]+-->/g, "")
      .replace(/\[\[ID:[^\]]+\]\]/g, "")
      .trim();

    const paperObj = { ID: id };
    const isAlreadySelected = id && isInSelectedNodeIDs && isInSelectedNodeIDs(id);
    const isAlreadyInSimilar = id && isInSimilarInputPapers && isInSimilarInputPapers(paperObj);
    const isAlreadySaved = id && isInSavedPapers && isInSavedPapers(paperObj);
    const isInQuestionList = selectedQuestionPapers.some(
      (p) => (p.id && p.id === id) || (!p.id && p.title === cleanTitle)
    );

    return (
      <div style={{ marginBottom: "0.5em" }}>
        <span
          style={{ color: "blue", fontWeight: "bold", cursor: "pointer" }}
          onClick={() => {
            Logger.logUIInteraction({
              component: "Dialog",
              action: "selectPaperFromResponse",
              paperTitle: title,
              previousSelectedPaper: chatSelectedPaper,
              paperId: id || undefined,
            });
            updateDialogState({ chatSelectedPaper: title });
            openPaperInfoModal(title, id);
          }}
        >
          {cleanTitle}
        </span>

        <div style={{ marginTop: "0.2em" }}>
          <DefaultButton
            iconProps={{ iconName: isInQuestionList ? "Cancel" : "Chat" }}
            text={isInQuestionList ? "" : "Ask"}
            styles={{ root: { marginRight: "0.3em", minWidth: 0 } }}
            onClick={() => toggleQuestionPaper({ id, title: cleanTitle })}
          />

          <DefaultButton
            iconProps={{ iconName: "Locate" }}
            styles={{ root: { marginRight: "0.3em", minWidth: 0 } }}
            disabled={isAlreadySelected}
            onClick={() => {
              if (id) {
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "locateSelectedPaper",
                  paperTitle: title,
                  paperId: id,
                });
                getPaperById(id).then((paper) => {
                  if (paper && paper["ID"] != null) {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "locateSelectedPaperSuccess",
                      paperTitle: title,
                      paperId: paper["ID"],
                      via: "id",
                    });
                    addToSelectNodeIDs([paper["ID"]], "scatterplot");
                  } else {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "locateSelectedPaperNotFound",
                      paperTitle: title,
                      paperId: id,
                      via: "id",
                    });
                  }
                });
              } else {
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "locateSelectedPaper",
                  paperTitle: title,
                });
                getPaperByTitle(title).then((papers) => {
                  if (papers.length > 0) {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "locateSelectedPaperSuccess",
                      paperTitle: title,
                      paperId: papers[0].ID,
                      papersFound: papers.length,
                      via: "title",
                    });
                    addToSelectNodeIDs(papers.map((d) => d["ID"]), "scatterplot");
                  } else {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "locateSelectedPaperNotFound",
                      paperTitle: title,
                      via: "title",
                    });
                  }
                });
              }
            }}
          />

          <DefaultButton
            iconProps={{ iconName: "PlusCircle" }}
            styles={{ root: { marginRight: "0.3em", minWidth: 0 } }}
            disabled={isAlreadyInSimilar}
            onClick={() => {
              if (id) {
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "addSelectedPaperToSimilarInput",
                  paperTitle: title,
                  paperId: id,
                });
                getPaperById(id).then((paper) => {
                  if (paper) {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "addSelectedPaperToSimilarInputSuccess",
                      paperTitle: title,
                      paperId: paper["ID"],
                    });
                    addToSimilarInputPapers(paper);
                  } else {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "addSelectedPaperToSimilarInputNotFound",
                      paperTitle: title,
                      paperId: id,
                    });
                  }
                });
              } else {
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "addSelectedPaperToSimilarInput",
                  paperTitle: title,
                });
                getPaperByTitle(title).then((papers) => {
                  if (papers.length > 0) {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "addSelectedPaperToSimilarInputSuccess",
                      paperTitle: title,
                      paperId: papers[0].ID,
                      papersFound: papers.length,
                      via: "title",
                    });
                    addToSimilarInputPapers(papers[0]);
                  } else {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "addSelectedPaperToSimilarInputNotFound",
                      paperTitle: title,
                    });
                  }
                });
              }
            }}
          />

          <DefaultButton
            iconProps={{ iconName: "Save" }}
            styles={{ root: { minWidth: 0 } }}
            disabled={isAlreadySaved}
            onClick={() => {
              if (id) {
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "saveSelectedPaper",
                  paperTitle: title,
                  paperId: id,
                });
                getPaperById(id).then((paper) => {
                  if (paper) {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "saveSelectedPaperSuccess",
                      paperTitle: title,
                      paperId: paper["ID"],
                    });
                    addToSavedPapers(paper);
                  } else {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "saveSelectedPaperNotFound",
                      paperTitle: title,
                      paperId: id,
                    });
                  }
                });
              } else {
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "saveSelectedPaper",
                  paperTitle: title,
                });
                getPaperByTitle(title).then((papers) => {
                  if (papers.length > 0) {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "saveSelectedPaperSuccess",
                      paperTitle: title,
                      paperId: papers[0].ID,
                      papersFound: papers.length,
                      via: "title",
                    });
                    addToSavedPapers(papers[0]);
                  } else {
                    Logger.logUIInteraction({
                      component: "Dialog",
                      action: "saveSelectedPaperNotFound",
                      paperTitle: title,
                    });
                  }
                });
              }
            }}
          />
        </div>
        {/* Render the extracted summary text */}
        {summary && (
          <div
            style={{
              color: "#333",
              fontSize: "0.95rem",
              lineHeight: "1.5em",
              whiteSpace: "pre-wrap",
              marginLeft: "0.2em",
              marginTop: "0.5em"
            }}
          >
            {summary}
          </div>
        )}
      </div>
    );
  };


  return (
    <div className="app-container" style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0, width: "100%", overflow: "hidden" }}>
      <style>{`@keyframes _agDot{0%,80%,100%{opacity:.2;transform:scale(.75)}40%{opacity:1;transform:scale(1.15)}}`}</style>

      {/* 1. SCROLLABLE AREA: Conversation (sidebar disabled) */}
      <div className="chat-content-row">
        {displayMessages.filter((m: any) => m.role !== "signal" && m.role !== "thinking").length === 0 && !isWaiting && !agentStatus ? (
          <div className="chat-empty-state" role="status" aria-live="polite">
            <p>Type below to start the conversation.</p>
          </div>
        ) : (
        <div className="chat-messages">
          {displayMessages.map((msg, idx) => {
            // Skip the thinking sentinel — it drives isWaiting only, not a visible bubble.
            if ((msg as any).role === "thinking") return null;

            // ── Signal cards (clarification / expansion) ──────────────────────
            if ((msg as any).role === "signal") {
              const sm = msg as any;
              const dismissSignal = () => updateDialogState((prev: any) => ({
                chatHistory: (prev.chatHistory || []).filter((_: any, i: number) => i !== idx),
              }));
              if (sm.signalType === "query_expansion") {
                const d = sm.signalData;
                return (
                  <div key={idx} style={{ background: "#fff", borderRadius: 16, boxShadow: "0 4px 24px rgba(0,0,0,0.10)", border: "1px solid #ebebeb", padding: "20px 24px", margin: "12px 0" }}>
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 4 }}>
                      <span style={{ flex: 1, fontSize: "0.975rem", fontWeight: 500, color: "#1a1a1a", lineHeight: 1.4 }}>
                        Which aspect of <em>"{d.original}"</em> are you looking for?
                      </span>
                      <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
                        <span style={{ fontSize: "0.78rem", color: "#aaa", whiteSpace: "nowrap" }}>{d.expansions.length} suggestions</span>
                        <IconButton iconProps={{ iconName: "Cancel" }} styles={{ root: { width: 28, height: 28, color: "#aaa" } }} title="Dismiss"
                          onClick={() => { dismissSignal(); setCustomExpansionText(null); }} />
                      </div>
                    </div>
                    {d.expansions.map((exp: string, i: number) => (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, padding: "11px 0", borderTop: "1px solid #f2f2f2", cursor: "pointer" }}
                        onClick={() => sendChat(exp)}
                        onMouseEnter={(e: React.MouseEvent<HTMLDivElement>) => (e.currentTarget.style.background = "#fafafa")}
                        onMouseLeave={(e: React.MouseEvent<HTMLDivElement>) => (e.currentTarget.style.background = "transparent")}>
                        <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#f0f0f0", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.82rem", fontWeight: 600, color: "#555", flexShrink: 0 }}>{i + 1}</div>
                        <span style={{ flex: 1, fontSize: "0.92rem", color: "#1a1a1a" }}>{exp}</span>
                        <span style={{ color: "#ccc", fontSize: "1.1rem" }}>›</span>
                      </div>
                    ))}
                    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "11px 0", borderTop: "1px solid #f2f2f2" }}>
                      <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#f0f0f0", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <span style={{ fontSize: "0.85rem", color: "#888" }}>✏</span>
                      </div>
                      {customExpansionText !== null ? (
                        <input autoFocus style={{ flex: 1, border: "none", outline: "none", fontSize: "0.92rem", color: "#1a1a1a", background: "transparent" }}
                          placeholder="Type your own query and press Enter…" value={customExpansionText}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCustomExpansionText(e.target.value)}
                          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                            if (e.key === "Enter" && customExpansionText.trim()) sendChat(customExpansionText);
                            if (e.key === "Escape") setCustomExpansionText(null);
                          }} />
                      ) : (
                        <span style={{ flex: 1, fontSize: "0.92rem", color: "#aaa", cursor: "text" }} onClick={() => setCustomExpansionText("")}>Something else</span>
                      )}
                      <DefaultButton text="Skip" styles={{ root: { minWidth: 0, borderRadius: 20, padding: "0 14px", height: 32, border: "1px solid #ddd" }, label: { fontSize: "0.82rem", color: "#555", fontWeight: 400 } }}
                        onClick={() => sendChat(d.original)} />
                    </div>
                  </div>
                );
              }
              if (sm.signalType === "intent_confirm") {
                const d = sm.signalData;
                return (
                  <div key={idx} style={{ background: "#fff", borderRadius: 16, boxShadow: "0 4px 24px rgba(0,0,0,0.10)", border: "1px solid #ebebeb", padding: "20px 24px", margin: "12px 0" }}>
                    <div style={{ display: "flex", alignItems: "flex-start", gap: 12, marginBottom: 4 }}>
                      <span style={{ flex: 1, fontSize: "0.975rem", fontWeight: 500, color: "#1a1a1a", lineHeight: 1.4 }}>Here's what I understood — please confirm before searching:</span>
                      <IconButton iconProps={{ iconName: "Cancel" }} styles={{ root: { width: 28, height: 28, color: "#aaa" } }} title="Dismiss"
                        onClick={() => { dismissSignal(); setCustomConfirmText(null); }} />
                    </div>
                    {d.items.map((item: { field: string; label: string; display: string; editable: string; confidence: number }, i: number) => (
                      <div key={i} style={{ display: "flex", alignItems: "flex-start", gap: 12, padding: "11px 0", borderTop: "1px solid #f2f2f2" }}>
                        <div style={{ width: 10, height: 10, borderRadius: "50%", marginTop: 5, flexShrink: 0, background: item.confidence >= 0.85 ? "#22c55e" : item.confidence >= 0.70 ? "#f59e0b" : "#ef4444" }} title={`Confidence: ${Math.round(item.confidence * 100)}%`} />
                        <div style={{ flex: 1 }}>
                          <div style={{ fontSize: "0.78rem", color: "#888", marginBottom: 2 }}>{item.label}</div>
                          <div style={{ fontSize: "0.92rem", color: "#1a1a1a" }}>{item.display}</div>
                        </div>
                        <DefaultButton text="Edit" styles={{ root: { minWidth: 0, borderRadius: 6, padding: "0 10px", height: 28, border: "1px solid #ddd", flexShrink: 0 }, label: { fontSize: "0.78rem", color: "#555", fontWeight: 400 } }}
                          onClick={() => { setDraftChatText(item.editable); dismissSignal(); textareaRef.current?.focus(); }} />
                      </div>
                    ))}
                    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "11px 0", borderTop: "1px solid #f2f2f2" }}>
                      <div style={{ width: 28, height: 28, borderRadius: "50%", background: "#f0f0f0", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                        <span style={{ fontSize: "0.85rem", color: "#888" }}>✏</span>
                      </div>
                      {customConfirmText !== null ? (
                        <input autoFocus style={{ flex: 1, border: "none", outline: "none", fontSize: "0.92rem", color: "#1a1a1a", background: "transparent" }}
                          placeholder="Rephrase your request and press Enter…" value={customConfirmText}
                          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setCustomConfirmText(e.target.value)}
                          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                            if (e.key === "Enter" && customConfirmText.trim()) sendChat(customConfirmText);
                            if (e.key === "Escape") setCustomConfirmText(null);
                          }} />
                      ) : (
                        <span style={{ flex: 1, fontSize: "0.92rem", color: "#aaa", cursor: "text" }} onClick={() => setCustomConfirmText("")}>Rephrase differently</span>
                      )}
                      <DefaultButton text="Confirm & Search" styles={{ root: { minWidth: 0, borderRadius: 20, padding: "0 14px", height: 32, background: "#0078d4", border: "none" }, label: { fontSize: "0.82rem", color: "#fff", fontWeight: 500 } }}
                        onClick={() => sendChat(d.original)} />
                    </div>
                  </div>
                );
              }
              return null;
            }

            const isLastAiMessage = msg.role === "ai" && idx === displayMessages.length - 1;
            return (
            <div key={idx} className={`chat-bubble ${msg.role}`}>
              {msg.role === "user" ? (
                <div>
                  {msg.referencedPapers && msg.referencedPapers.length > 0 && (
                    <div style={{ marginBottom: 6, display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {msg.referencedPapers.map((p) => (
                        <div
                          key={p.id || p.title}
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            padding: "2px 6px",
                            borderRadius: 10,
                            background: "var(--color-primary-light)",
                            color: "var(--color-primary)",
                            fontSize: "0.75rem",
                            maxWidth: "100%",
                          }}
                        >
                          <span
                            style={{
                              cursor: "pointer",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                              maxWidth: 220,
                            }}
                            onClick={() => openPaperInfoModal(p.title, p.id)}
                          >
                            {p.title}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  {msg.text}
                </div>
              ) : (
                <div>
                  {(() => {
                    const HAS_MORE_MARKER = "[[HAS_MORE_PAPERS]]";
                    const rawText = msg.text || "";
                    const refMap = buildRefMap(rawText);
                    const hasMarkerOrFlag = msg.hasMorePapers || rawText.includes(HAS_MORE_MARKER);
                    const looksLikePaperList = /Title:\s*/i.test(rawText) && /\[\[ID:[^\]]+\]\]/.test(rawText);
                    const hasLoadMore = hasMarkerOrFlag || (isLastAiMessage && looksLikePaperList);
                    const displayText = expandGroupedCitations(
                      rawText.replace(new RegExp("\\s*" + HAS_MORE_MARKER.replace(/[[\]]/g, "\\$&") + "\\s*", "g"), "").trim()
                    );
                    return (
                      <>
                        <Markdown
                          components={{
                            p: ({ node, children }) => {
                              const raw = extractText(children);
                              const { title, id } = parseTitleAndId(raw);
                              if (/Title:\s*/i.test(raw)) {
                                return renderPaperBlock(title, id, raw);
                              }
                              if (CITE_REGEX.test(raw)) {
                                CITE_REGEX.lastIndex = 0;
                                return <p>{renderInlineCitations(raw, refMap)}</p>;
                              }
                              return <p>{raw.replace(CITE_STRIP, "")}</p>;
                            },

                            strong: ({ node, ...props }) => {
                              const raw = extractText(props.children);
                              const { title, id } = parseTitleAndId(raw);
                              if (/Title:\s*/i.test(raw)) {
                                return renderPaperBlock(title, id, raw);
                              }
                              if (CITE_REGEX.test(raw)) {
                                CITE_REGEX.lastIndex = 0;
                                return <strong>{renderInlineCitations(raw, refMap)}</strong>;
                              }
                              return <strong>{raw.replace(CITE_STRIP, "")}</strong>;
                            },

                            li: ({ node, children }) => {
                              const raw = extractText(children);
                              const { title, id } = parseTitleAndId(raw);
                              if (/Title:\s*/i.test(raw)) {
                                return renderPaperBlock(title, id, raw);
                              }
                              if (CITE_REGEX.test(raw)) {
                                CITE_REGEX.lastIndex = 0;
                                return <li>{renderInlineCitations(raw, refMap)}</li>;
                              }
                              return <li>{raw.replace(CITE_STRIP, "")}</li>;
                            },

                          }}
                        >
                          {displayText}
                        </Markdown>
                        {hasLoadMore && (
                          <div style={{ marginTop: 8 }}>
                            <a
                              style={{ cursor: "pointer", color: "#0078d4", textDecoration: "underline" }}
                              onClick={() => sendChat("Load more papers from this search.")}
                            >
                              Load more papers
                            </a>
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
              )}
            </div>
          );
          })}

          {(isWaiting || agentStatus) && (
            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 4px" }}>
                <div style={{ display: "flex", gap: 5 }}>
                  {[0, 1, 2].map(i => (
                    <span key={i} style={{
                      display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: "#888",
                      animation: `_agDot 1.4s ease-in-out ${i * 0.18}s infinite both`,
                    }} />
                  ))}
                </div>
                <span style={{ fontSize: "0.88rem", color: "#666", fontStyle: "italic" }}>
                  {agentStatus || "Thinking..."}
                </span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
        )}

      </div>{/* end chat-content-row */}

      {/* 3. FIXED BOTTOM: Selected papers + input bar + action buttons */}
      <div className="chat-input-area">
          {selectedQuestionPapers.length > 0 && (
            <div style={{ marginBottom: 8, display: "flex", flexWrap: "wrap", gap: 6 }}>
              {selectedQuestionPapers.map((p) => (
                <div
                  key={p.id || p.title}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    padding: "4px 8px",
                    borderRadius: 12,
                    background: "var(--color-primary-light)",
                    color: "var(--color-primary)",
                    fontSize: "0.8rem",
                    maxWidth: "100%",
                  }}
                >
                  <span
                    style={{
                      cursor: "pointer",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      maxWidth: 220,
                    }}
                    onClick={() => openPaperInfoModal(p.title, p.id)}
                  >
                    {p.title}
                  </span>
                  <IconButton
                    styles={{ root: { marginLeft: 4, width: 20, height: 20 } }}
                    iconProps={{ iconName: "Cancel" }}
                    onClick={() => toggleQuestionPaper(p)}
                  />
                </div>
              ))}
            </div>
          )}

          <div className="chat-input-row">
            <textarea
              ref={textareaRef}
              className="chat-input"
              value={draftChatText}
              onChange={onChangeChatText}
              onKeyDown={onKeyDown}
              placeholder="Type your message"
              rows={1}
            />
            <DefaultButton
              className="iconButton"
              styles={{ root: { padding: "0 1rem", minWidth: 0 } }}
              onClick={() => {
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "askButtonClick",
                  value: draftChatText,
                  chatHistoryLength: chatHistory.length,
                });
                chatRequest();
              }}
              iconProps={{ iconName: "Rocket" }}
              text="Ask"
            />
          </div>

          <div className="chat-action-row" style={{ display: 'flex', flexWrap: 'nowrap', alignItems: 'center', gap: '4px' }}>
            <DefaultButton
              text="ALL"
              iconProps={{ iconName: "Locate" }}
              styles={{ root: { minWidth: 0, padding: "0 6px" } }}
              disabled={buttonsClicked.locateAll}
              onClick={async () => {
                const papers = extractAllPapers();
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "selectAllPapers",
                  responseLength: currentAIResponseText().length,
                  paperCount: papers.length,
                });

                if (papers.length === 0) {
                  console.log("No papers found in response");
                  return;
                }

                const paperIds: string[] = [];
                for (const paper of papers) {
                  if (paper.id) {
                    paperIds.push(paper.id);
                  }
                }

                if (paperIds.length > 0) {
                  addToSelectNodeIDs(paperIds, "scatterplot");
                  setButtonsClicked(prev => ({ ...prev, locateAll: true }));
                }
              }}
            />
            <DefaultButton
              text="ALL"
              iconProps={{ iconName: "PlusCircle" }}
              styles={{ root: { minWidth: 0, padding: "0 6px" } }}
              disabled={buttonsClicked.addAll}
              onClick={async () => {
                const papers = extractAllPapers();
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "addAllToSimilarInputPapers",
                  responseLength: currentAIResponseText().length,
                  paperCount: papers.length,
                });

                if (papers.length === 0) {
                  console.log("No papers found in response");
                  return;
                }

                for (const paper of papers) {
                  if (paper.id) {
                    try {
                      const paperObj = await getPaperById(paper.id);
                      if (paperObj) {
                        addToSimilarInputPapers(paperObj);
                      }
                    } catch (e) {
                      console.error(`Failed to fetch paper ${paper.id}:`, e);
                    }
                  }
                }
                setButtonsClicked(prev => ({ ...prev, addAll: true }));
              }}
            />
            <DefaultButton
              text="ALL"
              iconProps={{ iconName: "Save" }}
              styles={{ root: { minWidth: 0, padding: "0 6px" } }}
              disabled={buttonsClicked.saveAll}
              onClick={async () => {
                const papers = extractAllPapers();
                Logger.logUIInteraction({
                  component: "Dialog",
                  action: "saveAllPapers",
                  responseLength: currentAIResponseText().length,
                  paperCount: papers.length,
                });

                if (papers.length === 0) {
                  console.log("No papers found in response");
                  return;
                }

                for (const paper of papers) {
                  if (paper.id) {
                    try {
                      const paperObj = await getPaperById(paper.id);
                      if (paperObj) {
                        addToSavedPapers(paperObj);
                      }
                    } catch (e) {
                      console.error(`Failed to fetch paper ${paper.id}:`, e);
                    }
                  }
                }
                setButtonsClicked(prev => ({ ...prev, saveAll: true }));
              }}
            />
          </div>
        </div>

        {/* Paper Info Modal */}
        <Modal
          isOpen={isModalOpen}
          onDismiss={() => setModalState(false)}
          isBlocking={false}
          styles={{
            main: {
              maxWidth: 720,
              padding: 32,
              borderRadius: 16,
              boxShadow: "0 16px 40px rgba(0,0,0,0.25)",
            },
          }}
        >
          {loadingPaperInfo ? (
            <div style={{ padding: 20, textAlign: "center" }}>Loading details...</div>
          ) : paperInfo ? (
            <div style={{ maxHeight: "70vh", overflowY: "auto" }}>
              <h2 style={{ margin: 0, marginBottom: 12, fontSize: 24, lineHeight: 1.3 }}>
                {paperInfo.Title}
              </h2>
              <div style={{ fontSize: 14, lineHeight: 1.6, color: "#444", marginBottom: 12 }}>
                <b>Authors</b>:{" "}
                {Array.isArray(paperInfo.Authors)
                  ? paperInfo.Authors.join(", ")
                  : paperInfo.Authors || "N/A"}
                <br />
                <b>Source</b>: {paperInfo.Source || paperInfo.Venue || "N/A"}
                <br />
                <b>Year</b>: {paperInfo.Year || "N/A"}
                <br />
                <b>No. of Citations</b>: {paperInfo.CitationCounts || "N/A"}
                <br />
                <b>ID</b>: {paperInfo.ID || "N/A"}
              </div>
              <p>
                <b>Abstract</b>: {paperInfo.Abstract || "N/A"}
              </p>
              <div>
                <b>Keywords</b>:{" "}
                {Array.isArray(paperInfo.Keywords)
                  ? paperInfo.Keywords.join(", ")
                  : paperInfo.Keywords || "N/A"}
              </div>
              <hr style={{ margin: "20px 0" }} />
              <Stack horizontal tokens={{ childrenGap: 8 }}>
                <ActionButton
                  iconProps={{ iconName: "PlusCircle" }}
                  disabled={isInSimilarInputPapers && paperInfo ? isInSimilarInputPapers(paperInfo) : false}
                  onClick={() => {
                    if (paperInfo) {
                      Logger.logUIInteraction({
                        component: "Dialog",
                        action: "addToSimilarPapersFromModal",
                        paperId: paperInfo.ID,
                        paperTitle: paperInfo.Title,
                      });
                      addToSimilarInputPapers(paperInfo);
                      setModalState(false);
                    }
                  }}
                >
                  Select
                </ActionButton>
                <ActionButton
                  iconProps={{ iconName: "Save" }}
                  onClick={() => {
                    if (paperInfo) {
                      addToSavedPapers(paperInfo);
                      setModalState(false);
                    }
                  }}
                >
                  Save
                </ActionButton>
                <ActionButton
                  iconProps={{ iconName: "GraduationCap" }}
                  onClick={() => {
                    if (paperInfo && paperInfo.Title) {
                      const q = encodeURIComponent(paperInfo.Title);
                      window.open(`https://scholar.google.com/scholar?q=${q}`, "_blank", "noopener,noreferrer");
                    }
                  }}
                >
                  Google Scholar
                </ActionButton>
                <ActionButton
                  iconProps={{ iconName: "Cancel" }}
                  onClick={() => setModalState(false)}
                >
                  Close
                </ActionButton>
              </Stack>
            </div>
          ) : (
            <div style={{ padding: 20, textAlign: "center" }}>Paper not found</div>
          )}
        </Modal>

    </div>
  );
});


