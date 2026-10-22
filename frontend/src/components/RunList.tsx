import { useCallback, useEffect, useState } from "react";
import { listRuns, listStored } from "../api";
import { condLabel } from "./charts";
import { fmtDuration } from "./Controls";
import type { RunSummary, StoredStudy } from "../types";

interface RunListProps {
  activeRunId: string | null;
  onOpen: (runId: string) => void;
  /** Reload a study from disk (Storage tab), then open it. */
  onOpenStored: (runId: string) => void;
  /** Bumped by the parent to refresh immediately instead of waiting for a poll. */
  refreshToken: number;
}

function ago(ts: number | null): string {
  if (!ts) return "";
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

/**
 * Studies known to the backend, including ones this browser never started.
 *
 * Run state lives in the server process, not in the tab — so without this list
 * a second window had no way to reach a study already in flight, and a reload
 * lost it entirely.
 */
export default function RunList({ activeRunId, onOpen, onOpenStored, refreshToken }: RunListProps) {
  const [tab, setTab] = useState<"runs" | "storage">("runs");
  const [runs, setRuns] = useState<RunSummary[]>([]);

  const refresh = useCallback(() => {
    // A failed poll is not worth surfacing: the next tick retries, and the run
    // the user is actually watching has its own SSE stream for errors.
    listRuns()
      .then(setRuns)
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, refreshToken]);

  useEffect(() => {
    const id = window.setInterval(() => {
      if (!document.hidden) refresh(); // don't poll a backgrounded tab
    }, 4000);
    return () => window.clearInterval(id);
  }, [refresh]);

  return (
    <div className="run-list-wrap">
      <div className="tabs run-tabs">
        <button className={"tab" + (tab === "runs" ? " active" : "")} onClick={() => setTab("runs")}>
          Runs
        </button>
        <button className={"tab" + (tab === "storage" ? " active" : "")} onClick={() => setTab("storage")}>
          Storage
        </button>
      </div>
      {tab === "storage" ? (
        <StoredList activeRunId={activeRunId} onOpen={onOpenStored} refreshToken={refreshToken} />
      ) : runs.length === 0 ? (
        <p className="muted">No studies in memory. Saved ones are under Storage.</p>
      ) : (
      <>
      <p className="muted">
        Studies on this server — including any started in another window.
      </p>
      <ul className="run-list">
        {runs.map((r) => {
          // A finished study whose results were evicted is listed for the
          // record but can't be opened.
          const openable = r.status === "running" || r.available;
          const pct = r.progress.total
            ? (r.progress.current / r.progress.total) * 100
            : 0;
          return (
            <li
              key={r.run_id}
              className={
                "run-row" +
                (r.run_id === activeRunId ? " run-row-active" : "") +
                (openable ? "" : " run-row-dead")
              }
              onClick={() => openable && onOpen(r.run_id)}
              title={openable ? "Open this run" : "Results no longer held in memory"}
            >
              <div className="run-row-head">
                <span className={`run-badge run-badge-${r.status}`}>{r.status}</span>
                <span className="run-city">{r.city_label ?? r.city ?? "—"}</span>
                <span className="run-when">{ago(r.finished_at ?? r.created_at)}</span>
              </div>
              <div className="run-row-sub">
                {r.num_agents != null ? r.num_agents.toLocaleString() : "?"} agents ·{" "}
                {r.num_days}d · {r.num_seeds} seed{r.num_seeds === 1 ? "" : "s"} ·{" "}
                {r.conditions.length} condition{r.conditions.length === 1 ? "" : "s"}
                {r.reopened
                  ? " · reopened from storage"
                  : r.created_at && r.finished_at
                    ? ` · ran in ${fmtDuration(r.finished_at - r.created_at)}`
                    : ""}
              </div>
              {r.status === "running" && (
                <span className="run-track">
                  <span className="run-fill" style={{ width: `${pct}%` }} />
                </span>
              )}
              {!openable && <div className="run-row-sub run-dead">results expired — reopen it from Storage</div>}
              {r.status === "error" && r.error && (
                <div className="run-row-sub run-dead">{r.error}</div>
              )}
            </li>
          );
        })}
      </ul>
      </>
      )}
    </div>
  );
}

function fmtDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

/** The proximity weight(s) the study's built recommenders ran with, so studies
 *  made before and after a change to the Standard RS can be told apart. */
function proximityLabel(s: StoredStudy): string {
  const values = [...new Set(s.recommenders.filter((r) => !r.oracle).map((r) => r.w_proximity))];
  if (values.length === 0) return "";
  return `proximity ${values.map((v) => v.toFixed(2)).join(" / ")}`;
}

/**
 * Every study saved on disk. Studies are kept in server memory only while
 * recently used, and a restart drops them all, but each one's record and
 * simulation checkpoints stay on disk. Opening one reloads it from those
 * checkpoints exactly as it ran — nothing is re-simulated, so a study made
 * with an older recommender setting is still what it was.
 */
function StoredList({
  activeRunId,
  onOpen,
  refreshToken,
}: {
  activeRunId: string | null;
  onOpen: (runId: string) => void;
  refreshToken: number;
}) {
  const [studies, setStudies] = useState<StoredStudy[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listStored()
      .then((s) => {
        setStudies(s);
        setError(null);
      })
      .catch((e) => setError(e.message ?? String(e)));
  }, [refreshToken]);

  if (error) return <p className="error">{error}</p>;
  if (studies === null) return <p className="muted">Loading saved studies…</p>;
  if (studies.length === 0) return <p className="muted">No studies saved on disk yet.</p>;

  return (
    <>
      <p className="muted">
        Studies saved on disk. Opening one reloads it as it ran; nothing is re-simulated.
      </p>
      <ul className="run-list">
        {studies.map((s) => {
          const prox = proximityLabel(s);
          return (
            <li
              key={s.run_id}
              className={
                "run-row" +
                (s.run_id === activeRunId ? " run-row-active" : "") +
                (s.reopenable ? "" : " run-row-dead")
              }
              onClick={() => s.reopenable && onOpen(s.run_id)}
              title={s.reopenable ? "Open this study" : s.reason ?? "Can't be reopened"}
            >
              <div className="run-row-head">
                <span className="run-city">{s.city_label}</span>
                {s.open && <span className="run-badge run-badge-done">loaded</span>}
                <span className="run-when">{fmtDate(s.saved_at)}</span>
              </div>
              <div className="run-row-sub">
                {s.num_agents.toLocaleString()} agents · {s.num_days}d · {s.num_seeds} seed
                {s.num_seeds === 1 ? "" : "s"}
                {prox ? ` · ${prox}` : ""}
              </div>
              <table className="stored-t1">
                <tbody>
                  {s.table1.map((row) => (
                    <tr key={row.condition}>
                      <td>{condLabel(row.condition)}</td>
                      <td>{row.mean_utility.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {!s.reopenable && <div className="run-row-sub run-dead">results only: {s.reason}</div>}
            </li>
          );
        })}
      </ul>
    </>
  );
}
