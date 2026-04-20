import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Brain, Layers, Loader2, ScanEye, Sparkles, Timer } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

type PredictionResult = {
  predictions: Record<string, number>;
  predicted_class: string;
  confidence: number;
  uncertainity: boolean;
  warning: string | null;
  inference_time: number;
  filename?: string;
};

type SegmentationResult = {
  filename?: string;
  threshold: number;
  coverage: number;
  tumor_px: number;
  total_px: number;
  img_size: number;
  encoder: string;
  val_dice: number;
  timestamp: string;
  inference_time: number;
  mask_png: string;
  overlay_png: string;
  prob_png: string;
};

type Mode = 'detection' | 'segmentation';

const CLASS_ORDER = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] as const;

function clamp01(n: number) {
  return Math.max(0, Math.min(1, n));
}

function ConfidenceRing({ value }: { value: number }) {
  const v = clamp01(value);
  const pct = Math.round(v * 100);
  const r = 28;
  const c = 2 * Math.PI * r;
  const dash = c * v;
  const tone =
    v >= 0.85 ? 'stroke-emerald-500' : v >= 0.7 ? 'stroke-blue-600' : 'stroke-amber-500';

  return (
    <div className="relative grid h-16 w-16 place-items-center">
      <svg viewBox="0 0 80 80" className="h-16 w-16 -rotate-90">
        <circle cx="40" cy="40" r={r} className="fill-none stroke-black/10" strokeWidth="9" />
        <motion.circle
          cx="40"
          cy="40"
          r={r}
          className={`fill-none ${tone}`}
          strokeWidth="9"
          strokeLinecap="round"
          initial={{ strokeDasharray: `0 ${c}` }}
          animate={{ strokeDasharray: `${dash} ${c}` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute text-center">
        <div className="text-sm font-semibold text-zinc-950">{pct}%</div>
        <div className="text-[10px] font-medium text-black/50">conf</div>
      </div>
    </div>
  );
}

function UploadBox({
  disabled,
  file,
  previewUrl,
  onPick,
}: {
  disabled: boolean;
  file: File | null;
  previewUrl: string;
  onPick: (f: File) => void;
}) {
  const [dragging, setDragging] = useState(false);

  return (
    <div
      className={[
        'rounded-2xl border border-dashed p-5 transition',
        dragging
          ? 'border-blue-400 bg-gradient-to-br from-blue-50 via-white to-indigo-50'
          : 'border-black/15 bg-white hover:bg-gradient-to-br hover:from-white hover:to-blue-50',
        disabled ? 'opacity-60' : '',
      ].join(' ')}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (disabled) return;
        const f = e.dataTransfer.files?.[0];
        if (f) onPick(f);
      }}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-sm font-semibold text-zinc-950">Upload MRI scan</div>
          <div className="mt-1 text-xs text-black/55">JPG/PNG • drag & drop supported</div>
        </div>
        {file && (
          <div className="max-w-[60%] truncate rounded-full border border-blue-100 bg-blue-50 px-3 py-1 text-xs font-semibold text-blue-800">
            {file.name}
          </div>
        )}
      </div>

      <div className="mt-4">
        {previewUrl ? (
          <div className="overflow-hidden rounded-xl border border-black/10 bg-zinc-50">
            <img src={previewUrl} alt="Preview" className="h-56 w-full object-cover" />
          </div>
        ) : (
          <label className="block cursor-pointer rounded-xl border border-black/10 bg-white px-4 py-4 text-center shadow-sm hover:shadow-md transition">
            <input
              className="hidden"
              type="file"
              accept="image/png,image/jpeg"
              disabled={disabled}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onPick(f);
              }}
            />
            <div className="mx-auto grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-blue-100 to-indigo-100 ring-1 ring-blue-200/70">
              <Brain className="h-6 w-6 text-indigo-700" />
            </div>
            <div className="mt-3 text-sm font-semibold">Drag & drop your MRI image</div>
            <div className="mt-1 text-sm text-black/60">or click to browse</div>
          </label>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState<Mode>('detection');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [seg, setSeg] = useState<SegmentationResult | null>(null);
  const [segThreshold, setSegThreshold] = useState(0.5);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showProb, setShowProb] = useState(true);

  const canAnalyze = useMemo(() => !!file && !loading, [file, loading]);

  const pickFile = (f: File) => {
    const ok = f.type === 'image/png' || f.type === 'image/jpeg' || f.type === 'image/jpg';
    if (!ok) {
      setError('Only JPG/PNG images are supported.');
      return;
    }
    setFile(f);
    setResult(null);
    setSeg(null);
    setError('');
    const r = new FileReader();
    r.onloadend = () => setPreviewUrl(String(r.result ?? ''));
    r.readAsDataURL(f);
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl('');
    setResult(null);
    setSeg(null);
    setError('');
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    setResult(null);
    setSeg(null);

    try {
      const fd = new FormData();
      fd.append('file', file);
      const url =
        mode === 'segmentation'
          ? `http://localhost:8000/segment?threshold=${encodeURIComponent(String(segThreshold))}`
          : 'http://localhost:8000/predict';

      const res = await fetch(url, { method: 'POST', body: fd });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || 'Failed to analyze image');
      }
      const json = await res.json();
      if (mode === 'segmentation') setSeg(json as SegmentationResult);
      else setResult(json as PredictionResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  const chartData =
    result &&
    CLASS_ORDER.map((k) => ({
      name: k.replace('_', ' '),
      value: Math.round(((result.predictions?.[k] ?? 0) as number) * 100),
    }));

  return (
    <div className="hidden lg:block">
      <div className="mx-auto max-w-[1200px] px-10 py-12">
        <div className="rounded-[28px] border border-black/10 bg-white shadow-[0_24px_70px_-48px_rgba(37,99,235,0.45)] overflow-hidden">
          <div className="h-1.5 w-full bg-gradient-to-r from-blue-600 via-sky-500 to-indigo-600" />
          <div className="border-b border-black/10 px-10 py-7 bg-gradient-to-b from-white to-blue-50/30">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-semibold tracking-tight text-zinc-950">
                  Brain MRI AI Suite
                </div>
                <div className="mt-1 text-sm text-black/60">
                  Choose detection or segmentation from the same upload
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="inline-flex rounded-full border border-black/10 bg-white p-1 shadow-sm">
                  <button
                    type="button"
                    onClick={() => {
                      setMode('detection');
                      setResult(null);
                      setSeg(null);
                      setError('');
                    }}
                    className={[
                      'inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold transition',
                      mode === 'detection'
                        ? 'bg-gradient-to-r from-blue-600 via-sky-600 to-indigo-600 text-white shadow-sm'
                        : 'text-black/70 hover:bg-black/5',
                    ].join(' ')}
                  >
                    <ScanEye className="h-4 w-4" />
                    Detection
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setMode('segmentation');
                      setResult(null);
                      setSeg(null);
                      setError('');
                    }}
                    className={[
                      'inline-flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold transition',
                      mode === 'segmentation'
                        ? 'bg-gradient-to-r from-blue-600 via-sky-600 to-indigo-600 text-white shadow-sm'
                        : 'text-black/70 hover:bg-black/5',
                    ].join(' ')}
                  >
                    <Layers className="h-4 w-4" />
                    Segmentation
                  </button>
                </div>
                <div className="rounded-full border border-blue-200 bg-white px-4 py-2 text-xs font-semibold text-blue-700 shadow-sm">
                  Desktop • Medical UI
                </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-12 gap-8 px-10 py-10">
            <div className="col-span-5">
              <UploadBox disabled={loading} file={file} previewUrl={previewUrl} onPick={pickFile} />

              {mode === 'segmentation' && (
                <div className="mt-4 rounded-2xl border border-black/10 bg-white p-5">
                  <div className="text-sm font-semibold text-zinc-950">Segmentation settings</div>
                  <div className="mt-3 grid grid-cols-12 gap-4 items-center">
                    <div className="col-span-4 text-xs font-semibold text-black/60">Threshold</div>
                    <div className="col-span-8">
                      <input
                        type="range"
                        min={0.1}
                        max={0.9}
                        step={0.05}
                        value={segThreshold}
                        onChange={(e) => setSegThreshold(Number(e.target.value))}
                        className="w-full accent-blue-600"
                      />
                      <div className="mt-1 flex items-center justify-between text-xs text-black/50">
                        <span>0.10</span>
                        <span className="font-semibold text-black/70">{segThreshold.toFixed(2)}</span>
                        <span>0.90</span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 flex items-center gap-4">
                    <label className="inline-flex items-center gap-2 text-xs font-semibold text-black/70">
                      <input
                        type="checkbox"
                        checked={showProb}
                        onChange={(e) => setShowProb(e.target.checked)}
                        className="accent-blue-600"
                      />
                      Probability map
                    </label>
                    <label className="inline-flex items-center gap-2 text-xs font-semibold text-black/70">
                      <input
                        type="checkbox"
                        checked={showOverlay}
                        onChange={(e) => setShowOverlay(e.target.checked)}
                        className="accent-blue-600"
                      />
                      Overlay
                    </label>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-4 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
                  {error}
                </div>
              )}

              <div className="mt-4 flex gap-3">
                <button
                  type="button"
                  onClick={analyze}
                  disabled={!canAnalyze}
                  className="inline-flex flex-1 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-blue-600 via-sky-600 to-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Sparkles className="h-5 w-5" />}
                  {loading ? 'Analyzing…' : mode === 'segmentation' ? 'Segment MRI' : 'Analyze MRI'}
                </button>
                <button
                  type="button"
                  onClick={reset}
                  disabled={!file && !result && !seg}
                  className="rounded-2xl border border-black/10 bg-white px-5 py-3 text-sm font-semibold text-black/70 shadow-sm transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Reset
                </button>
              </div>
            </div>

            <div className="col-span-7">
              <div className="rounded-2xl border border-black/10 bg-white p-6">
                <div className="text-sm font-semibold text-zinc-950">Results</div>
                <div className="mt-1 text-sm text-black/55">
                  {mode === 'segmentation'
                    ? 'Mask, overlay, probability map, and coverage metrics'
                    : 'Predicted class, confidence, inference time, and probabilities'}
                </div>

                {mode === 'segmentation' ? (
                  !seg ? (
                    <div className="mt-6 rounded-2xl border border-black/10 bg-zinc-50 p-8 text-sm text-black/60">
                      Upload an image and click <span className="font-semibold text-black/75">Segment MRI</span>.
                    </div>
                  ) : (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.25, ease: 'easeOut' }}
                      className="mt-6 space-y-4"
                    >
                      <div className="rounded-2xl border border-black/10 bg-white p-6">
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="text-sm text-black/55">Tumor coverage</div>
                            <div className="mt-1 text-4xl font-semibold tracking-tight text-zinc-950">
                              {seg.coverage.toFixed(1)}%
                            </div>
                            <div className="mt-3 flex items-center gap-2 text-sm text-black/55">
                              <Timer className="h-4 w-4" />
                              <span>
                                Inference time:{' '}
                                <span className="font-semibold text-black/75">{seg.inference_time.toFixed(1)} ms</span>
                              </span>
                            </div>
                            <div className="mt-2 text-xs text-black/50">
                              Encoder: <span className="font-semibold text-black/70">{seg.encoder}</span> • Size:{' '}
                              <span className="font-semibold text-black/70">{seg.img_size}×{seg.img_size}</span> • Val
                              Dice: <span className="font-semibold text-black/70">{seg.val_dice}</span>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-xs font-semibold text-black/60">Threshold</div>
                            <div className="mt-1 text-2xl font-semibold text-zinc-950">{seg.threshold.toFixed(2)}</div>
                            <div className="mt-2 text-xs text-black/50">
                              Tumor pixels:{' '}
                              <span className="font-semibold text-black/70">{seg.tumor_px.toLocaleString()}</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-12 gap-4">
                        <div className="col-span-6 rounded-2xl border border-black/10 bg-white p-4">
                          <div className="text-sm font-semibold text-zinc-950">Mask</div>
                          <div className="mt-3 overflow-hidden rounded-xl border border-black/10 bg-zinc-50">
                            <img
                              src={`data:image/png;base64,${seg.mask_png}`}
                              alt="Mask"
                              className="h-56 w-full object-contain"
                            />
                          </div>
                        </div>
                        {showOverlay && (
                          <div className="col-span-6 rounded-2xl border border-black/10 bg-white p-4">
                            <div className="text-sm font-semibold text-zinc-950">Overlay</div>
                            <div className="mt-3 overflow-hidden rounded-xl border border-black/10 bg-zinc-50">
                              <img
                                src={`data:image/png;base64,${seg.overlay_png}`}
                                alt="Overlay"
                                className="h-56 w-full object-contain"
                              />
                            </div>
                          </div>
                        )}
                        {showProb && (
                          <div className="col-span-12 rounded-2xl border border-black/10 bg-white p-4">
                            <div className="text-sm font-semibold text-zinc-950">Probability map</div>
                            <div className="mt-3 overflow-hidden rounded-xl border border-black/10 bg-zinc-50">
                              <img
                                src={`data:image/png;base64,${seg.prob_png}`}
                                alt="Probability"
                                className="h-64 w-full object-contain"
                              />
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )
                ) : !result ? (
                  <div className="mt-6 rounded-2xl border border-black/10 bg-zinc-50 p-8 text-sm text-black/60">
                    Upload an image and click <span className="font-semibold text-black/75">Analyze MRI</span>.
                  </div>
                ) : (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25, ease: 'easeOut' }}
                    className="mt-6 space-y-4"
                  >
                    {(result.uncertainity || result.warning) && (
                      <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
                        <div className="flex items-start gap-3">
                          <AlertTriangle className="mt-0.5 h-5 w-5 text-amber-700" />
                          <div>
                            <div className="font-semibold">Warning</div>
                            <div className="mt-1 text-amber-900/80">
                              {result.warning ??
                                'Prediction is uncertain. Please consult a medical professional.'}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="rounded-2xl border border-black/10 bg-white p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="text-sm text-black/55">Predicted class</div>
                          <div className="mt-1 text-4xl font-semibold capitalize tracking-tight text-zinc-950">
                            {result.predicted_class.replace('_', ' ')}
                          </div>
                          <div className="mt-3 flex items-center gap-2 text-sm text-black/55">
                            <Timer className="h-4 w-4" />
                            <span>
                              Inference time:{' '}
                              <span className="font-semibold text-black/75">{result.inference_time.toFixed(1)} ms</span>
                            </span>
                          </div>
                        </div>
                        <ConfidenceRing value={result.confidence} />
                      </div>
                    </div>

                    <div className="rounded-2xl border border-black/10 bg-white p-6">
                      <div className="text-sm font-semibold text-zinc-950">Probability (all classes)</div>
                      <div className="mt-4 h-[260px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={chartData ?? []} margin={{ top: 8, right: 10, left: -10, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.08)" />
                            <XAxis
                              dataKey="name"
                              tick={{ fill: 'rgba(0,0,0,0.65)', fontSize: 12 }}
                              axisLine={{ stroke: 'rgba(0,0,0,0.15)' }}
                              tickLine={false}
                            />
                            <YAxis
                              tick={{ fill: 'rgba(0,0,0,0.6)', fontSize: 12 }}
                              axisLine={{ stroke: 'rgba(0,0,0,0.15)' }}
                              tickLine={false}
                              domain={[0, 100]}
                              unit="%"
                            />
                            <Tooltip
                              cursor={{ fill: 'rgba(59,130,246,0.08)' }}
                              formatter={(v) => [`${Number(v ?? 0)}%`, 'Probability']}
                              contentStyle={{
                                borderRadius: 14,
                                border: '1px solid rgba(0,0,0,0.08)',
                                boxShadow: '0 12px 40px -22px rgba(0,0,0,0.35)',
                              }}
                            />
                            <Bar
                              dataKey="value"
                              radius={[10, 10, 10, 10]}
                              fill="url(#barGrad)"
                              isAnimationActive
                              animationDuration={700}
                            />
                            <defs>
                              <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#2563eb" stopOpacity={0.95} />
                                <stop offset="55%" stopColor="#0ea5e9" stopOpacity={0.85} />
                                <stop offset="100%" stopColor="#4f46e5" stopOpacity={0.70} />
                              </linearGradient>
                            </defs>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
