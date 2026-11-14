import { useState, useMemo, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { X, ArrowRight, Loader2, Plus, Minus } from 'lucide-react'
import { api, type ScreenerStrategy } from '@/lib/api'
import { QK } from '@/lib/queryKeys'

interface Props {
  open: boolean
  onClose: () => void
  onSavedId?: (id: string) => void | Promise<void>
  /** 编辑模式: 传入已有 sequential 策略 id 时为 update */
  editStrategyId?: string | null
}

const MAX_GAP = 60
const MIN_STAGES = 2
const MAX_STAGES = 4

export function SequentialStrategyDialog({ open, onClose, onSavedId, editStrategyId }: Props) {
  const isEdit = !!editStrategyId
  const [name, setName] = useState('')
  const [nameDirty, setNameDirty] = useState(false)
  const [description, setDescription] = useState('')
  const [strategyId, setStrategyId] = useState('')
  const [stages, setStages] = useState<string[]>(['', ''])
  const [minGaps, setMinGaps] = useState<number[]>([0])
  const [maxGaps, setMaxGaps] = useState<number[]>([3])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const mouseDownOnBackdrop = useRef(false)

  const availableQuery = useQuery({
    queryKey: QK.screenerStrategies('all'),
    queryFn: () => api.screenerStrategies(),
    enabled: open,
  })

  // 子策略只能是日线叶子策略: 排除叠加/顺序 (source=composite) 与分钟策略
  const available = useMemo(
    () =>
      (availableQuery.data?.presets ?? []).filter(
        s => s.source !== 'composite' && !s.timeframes?.includes('1m'),
      ),
    [availableQuery.data],
  )

  useEffect(() => {
    if (!open) return
    setName('')
    setDescription('')
    setStrategyId(isEdit ? (editStrategyId ?? '') : `sequential_${Date.now().toString(36)}`)
    setStages(['', ''])
    setMinGaps([0])
    setMaxGaps([3])
    setError('')
    setNameDirty(isEdit)
    if (isEdit && editStrategyId) {
      api
        .strategyGet(editStrategyId)
        .then(detail => {
          setName(detail.name)
          setDescription(detail.description)
          const seq = detail.sequential
          if (seq) {
            const loadedStages = seq.stages?.length ? seq.stages : [seq.pre, seq.trigger]
            setStages(loadedStages)
            setMinGaps(seq.min_gaps?.length ? seq.min_gaps : [0])
            setMaxGaps(seq.max_gaps?.length ? seq.max_gaps : [seq.max_gap_days ?? 3])
          }
        })
        .catch(() => {})
    }
  }, [open, isEdit, editStrategyId])

  // 未手动命名时自动生成: 各段策略名用 → 连接
  useEffect(() => {
    if (nameDirty) return
    const names = stages
      .map(id => available.find(s => s.id === id)?.name ?? id)
      .filter(Boolean)
    setName(names.join(' → '))
  }, [stages, available, nameDirty])

  const setStage = (index: number, value: string) => {
    setStages(prev => prev.map((item, i) => (i === index ? value : item)))
  }

  const addStage = () => {
    if (stages.length >= MAX_STAGES) return
    setStages(prev => [...prev, ''])
    setMinGaps(prev => [...prev, 0])
    setMaxGaps(prev => [...prev, 3])
  }

  const removeStage = (index: number) => {
    if (stages.length <= MIN_STAGES) return
    setStages(prev => prev.filter((_, i) => i !== index))
    // 删除某段后: 该段前的窗口与后一段窗口合并, 简单保留前一段窗口
    setMinGaps(prev => prev.filter((_, i) => i !== Math.min(index, prev.length - 1)))
    setMaxGaps(prev => prev.filter((_, i) => i !== Math.min(index, prev.length - 1)))
  }

  const updateGap = (index: number, field: 'min' | 'max', value: number) => {
    const clamped = Math.max(0, Math.min(MAX_GAP, value || 0))
    if (field === 'min') {
      setMinGaps(prev => prev.map((item, i) => (i === index ? clamped : item)))
    } else {
      setMaxGaps(prev => prev.map((item, i) => (i === index ? clamped : item)))
    }
  }

  const handleSave = async () => {
    if (stages.some(id => !id)) {
      setError('请为每一段选择策略')
      return
    }
    if (new Set(stages).size !== stages.length) {
      setError('同一策略不能重复出现在多个段')
      return
    }
    for (let i = 0; i < minGaps.length; i += 1) {
      if (minGaps[i] > maxGaps[i]) {
        setError(`第 ${i + 1} 段间隔：最小不能大于最大`)
        return
      }
    }
    setSaving(true)
    setError('')
    try {
      const res = await api.strategySaveSequential({
        strategy_id: strategyId,
        name: name || strategyId,
        description,
        stages,
        min_gaps: minGaps,
        max_gaps: maxGaps,
        mode: isEdit ? 'update' : 'create',
      })
      await onSavedId?.(res.strategy_id)
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  const stageOptions = (index: number) => {
    const used = new Set(stages.filter((_, i) => i !== index && stages[i]))
    return available.filter(s => !used.has(s.id))
  }

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onMouseDown={e => {
            mouseDownOnBackdrop.current = e.target === e.currentTarget
          }}
          onClick={e => {
            if (mouseDownOnBackdrop.current && e.target === e.currentTarget) onClose()
          }}
        >
          <motion.div
            className="relative flex max-h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-panel border border-border bg-base shadow-2xl"
            initial={{ scale: 0.96, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.96, opacity: 0 }}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <ArrowRight className="h-4 w-4 text-cyan-400" />
              <span className="text-sm font-semibold text-foreground">
                {isEdit ? '编辑顺序策略' : '创建顺序策略'}
              </span>
              <span className="text-[10px] text-muted/60">
                最多 4 段，相邻段可设最小/最大间隔
              </span>
              <button onClick={onClose} className="ml-auto text-muted hover:text-foreground">
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="flex-1 space-y-4 overflow-y-auto p-4">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div>
                  <label className="mb-1 block text-xs text-muted">策略 ID（自动生成）</label>
                  <input
                    value={strategyId}
                    readOnly
                    disabled
                    className="w-full cursor-not-allowed rounded-btn border border-border bg-elevated px-2 py-1.5 font-mono text-[11px] text-muted"
                  />
                </div>
                <div>
                  <label className="mb-1 block text-xs text-muted">策略名称</label>
                  <input
                    value={name}
                    onChange={e => {
                      setName(e.target.value)
                      setNameDirty(true)
                    }}
                    placeholder="蓄势 → 突破 → 回踩"
                    className="w-full rounded-btn border border-border bg-elevated px-2 py-1.5 text-xs text-foreground placeholder:text-muted/40"
                  />
                </div>
              </div>

              <div>
                <label className="mb-1 block text-xs text-muted">描述（可选）</label>
                <input
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="说明这个顺序策略的意图"
                  className="w-full rounded-btn border border-border bg-elevated px-2 py-1.5 text-xs text-foreground placeholder:text-muted/40"
                />
              </div>

              <div className="space-y-3 rounded-btn border border-border bg-elevated/40 p-3">
                {stages.map((stageId, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-cyan-500/15 px-1.5 py-0.5 text-[11px] font-medium text-cyan-300">
                        {index + 1}
                      </span>
                      <select
                        value={stageId}
                        onChange={e => setStage(index, e.target.value)}
                        className="flex-1 rounded-btn border border-border bg-elevated px-2 py-1.5 text-xs text-foreground"
                      >
                        <option value="">请选择策略…</option>
                        {stageOptions(index).map((s: ScreenerStrategy) => (
                          <option key={s.id} value={s.id}>
                            {s.name}（{s.id}）
                          </option>
                        ))}
                      </select>
                      {stages.length > MIN_STAGES && (
                        <button
                          onClick={() => removeStage(index)}
                          title="删除该段"
                          className="text-muted/60 hover:text-danger"
                        >
                          <Minus className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>

                    {index < stages.length - 1 && (
                      <div className="flex items-center gap-2 pl-7 text-[11px] text-muted">
                        <span>间隔</span>
                        <input
                          type="number"
                          min={0}
                          max={MAX_GAP}
                          value={minGaps[index] ?? 0}
                          onChange={e => updateGap(index, 'min', parseInt(e.target.value) || 0)}
                          className="w-16 rounded-btn border border-border bg-base px-2 py-1 text-xs text-foreground"
                          aria-label={`第${index + 1}段最小间隔`}
                        />
                        <span>~</span>
                        <input
                          type="number"
                          min={0}
                          max={MAX_GAP}
                          value={maxGaps[index] ?? 3}
                          onChange={e => updateGap(index, 'max', parseInt(e.target.value) || 0)}
                          className="w-16 rounded-btn border border-border bg-base px-2 py-1 text-xs text-foreground"
                          aria-label={`第${index + 1}段最大间隔`}
                        />
                        <span>个交易日</span>
                      </div>
                    )}
                  </div>
                ))}

                {stages.length < MAX_STAGES && (
                  <button
                    onClick={addStage}
                    className="inline-flex items-center gap-1 rounded-btn border border-dashed border-border px-2 py-1 text-[11px] text-muted hover:border-cyan-500/40 hover:text-cyan-300"
                  >
                    <Plus className="h-3 w-3" />
                    增加一段
                  </button>
                )}
              </div>

              <div className="rounded-btn border border-border/60 bg-elevated/30 px-3 py-2 text-[11px] leading-relaxed text-muted">
                判定：从第 1 段开始逐段推进，相邻段命中日必须落在各自的最小~最大间隔内（交易日）。
                同一天可连续推进（间隔 0，需最小间隔为 0）。最后一段命中日即入选日，评分取最后一段，
                结果附带每段实际间隔与总间隔。
              </div>

              {error && (
                <div className="rounded-btn border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                  {error}
                </div>
              )}
            </div>

            <div className="flex items-center justify-end gap-2 border-t border-border px-4 py-3">
              <button
                onClick={onClose}
                className="rounded-btn border border-border px-3 py-1.5 text-xs text-muted hover:text-foreground"
              >
                取消
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="inline-flex items-center gap-1.5 rounded-btn border border-cyan-500/30 bg-cyan-500/15 px-3 py-1.5 text-xs font-medium text-cyan-300 hover:bg-cyan-500/25 disabled:opacity-50"
              >
                {saving && <Loader2 className="h-3 w-3 animate-spin" />}
                {isEdit ? '保存修改' : '创建'}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
