import { useEffect, useMemo, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { HexagonLayer } from "@deck.gl/aggregation-layers";
import { condLabel } from "./charts";
import type { RunMeta, WelfareMapRow } from "../types";

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

/**
 * Where a recommender's winners and losers live.
 *
 * Each agent contributes its home point and the change in its mean leisure net
 * utility versus the No-RS control. Hexagons average those deltas, so a cell
 * answers "for people who live here, did this recommender help or hurt".
 *
 * Any two conditions can be compared. Every delta is already relative to the
 * control, so the control's own delta is zero and A-versus-B is delta[A] −
 * delta[B] for agents who went out under both.
 *
 * Averaged, never summed: a hexagon over a dense residential block holds far
 * more agents than one in the outer counties, and a sum would render population
 * density as welfare impact.
 */

// Diverging orange→blue. Mid is near-transparent so unaffected areas recede and
// the eye goes to the extremes, which is the whole point of the panel.
const COLOR_RANGE: [number, number, number][] = [
  [180, 78, 12],
  [214, 130, 62],
  [235, 190, 150],
  [170, 195, 230],
  [86, 138, 210],
  [28, 82, 172],
];

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

interface MapPoint {
  lat: number;
  lon: number;
  value: number;
}

export default function WelfareMap({
  run,
  conditions,
  control,
}: {
  run: RunMeta;
  conditions: string[];
  control: string;
}) {
  const [conditionPick, setCondition] = useState(
    conditions.find((c) => c !== control) ?? control
  );
  const [baselinePick, setBaseline] = useState(control);
  const condition = conditions.includes(conditionPick)
    ? conditionPick
    : conditions.find((c) => c !== control) ?? control;
  const baseline = conditions.includes(baselinePick) ? baselinePick : control;
  const [radius, setRadius] = useState(600);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);

  const rows: WelfareMapRow[] = run.welfare_map ?? [];
  const points = useMemo<MapPoint[]>(() => {
    const deltaOf = (r: WelfareMapRow, c: string) => (c === control ? 0 : r.delta[c]);
    const out: MapPoint[] = [];
    for (const r of rows) {
      const a = deltaOf(r, condition);
      const b = deltaOf(r, baseline);
      if (a != null && b != null) out.push({ lat: r.lat, lon: r.lon, value: a - b });
    }
    return out;
  }, [rows, condition, baseline, control]);

  // Symmetric colour domain from the 5th/95th percentile of |delta|, so a
  // single outlier cannot flatten the whole map to one shade, and zero always
  // sits at the palette's midpoint (otherwise "blue" would not mean "better").
  const domain = useMemo<[number, number]>(() => {
    const values = points.map((p) => Math.abs(p.value)).sort((a, b) => a - b);
    const extent = Math.max(1e-4, quantile(values, 0.95));
    return [-extent, extent];
  }, [points]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: MAP_STYLE,
      center: [run.view.longitude, run.view.latitude],
      zoom: 8.6,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    const overlay = new MapboxOverlay({ interleaved: false, layers: [] });
    map.addControl(overlay as unknown as maplibregl.IControl);
    mapRef.current = map;
    overlayRef.current = overlay;
    return () => {
      overlayRef.current = null;
      mapRef.current = null;
      map.remove();
    };
  }, [run.view.latitude, run.view.longitude]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    overlay.setProps({
      layers: [
        new HexagonLayer<MapPoint>({
          id: `welfare-${condition}-${baseline}-${radius}`,
          data: points,
          getPosition: (d) => [d.lon, d.lat],
          getColorWeight: (d) => d.value,
          colorAggregation: "MEAN",
          radius,
          coverage: 0.92,
          extruded: false,
          opacity: 0.75,
          colorRange: COLOR_RANGE,
          colorDomain: domain,
          pickable: true,
        }),
      ],
      getTooltip: ({ object }: { object?: { points?: unknown[]; colorValue?: number } }) => {
        if (!object) return null;
        const n = object.points?.length ?? 0;
        const mean = object.colorValue ?? 0;
        return {
          html: `<b>${mean >= 0 ? "+" : ""}${mean.toFixed(3)}</b> mean ΔU<br/>${n} agent${n === 1 ? "" : "s"} living here`,
          style: { fontSize: "12px" },
        };
      },
    });
  }, [points, condition, baseline, radius, domain]);

  if (!rows.length) {
    return (
      <p className="muted small">
        No paired agents to map — this needs agents who took a leisure trip both
        with and without a recommender.
      </p>
    );
  }

  return (
    <div className="welfare-map">
      <div className="welfare-map-controls">
        {conditions.length > 1 && (
          <>
            <label className="inline-picker">
              Condition
              <select value={condition} onChange={(e) => setCondition(e.target.value)}>
                {conditions.map((c) => (
                  <option key={c} value={c}>{condLabel(c)}</option>
                ))}
              </select>
            </label>
            <label className="inline-picker">
              Versus
              <select value={baseline} onChange={(e) => setBaseline(e.target.value)}>
                {conditions.map((c) => (
                  <option key={c} value={c}>{condLabel(c)}</option>
                ))}
              </select>
            </label>
          </>
        )}
        <label className="inline-picker">
          Hex size
          <select value={radius} onChange={(e) => setRadius(parseInt(e.target.value, 10))}>
            <option value={300}>300 m</option>
            <option value={600}>600 m</option>
            <option value={1200}>1.2 km</option>
            <option value={2500}>2.5 km</option>
          </select>
        </label>
        <span className="muted small">
          {points.length.toLocaleString()} agents mapped
          {condition === baseline ? " · same condition on both sides, so every cell is zero" : ""}
        </span>
      </div>
      <div ref={containerRef} className="welfare-map-canvas" />
      <div className="welfare-legend">
        <span className="muted small">worse off</span>
        <span className="legend-ramp" />
        <span className="muted small">better off</span>
        <span className="muted small">
          under {condLabel(condition)} vs {condLabel(baseline)}
        </span>
        <span className="muted small legend-range">
          ±{Math.abs(domain[1]).toFixed(2)} utility
        </span>
      </div>
    </div>
  );
}
