import { useState, useMemo, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { X, ArrowRight, Loader2 } from 'lucide-react'
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

export function SequentialStrategyDialog({ open, onClose, onSavedId, editStrategyId }: Props) {
  const isEdit = !!editStrategyId
  const [name, setName] = useState('')
  const [nameDirty, setNameDirty] = useState(false)
  const [description, setDescription] = useState('')
  const [strategyId, setStrategyId] = useState('')
  const [preId, setPreId] = useState('')
  const [triggerId, setTriggerId] = useState('')
  const [maxGap, setMaxGap] = useState(3)
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
    setPreId('')
    setTriggerId('')
    setMaxGap(3)
    setError('')
    setNameDirty(isEdit)
    if (isEdit && editStrategyId) {
      api
        .strategyGet(editStrategyId)
        .then(detail => {
          setName(detail.name)
          setDescription(detail.description)
          if (detail.sequential) {
            setPreId(detail.sequential.pre)
            setTriggerId(detail.sequential.trigger)
            setMaxGap(detail.sequential.max_gap_days)
          }
        })
        .catch(() => {})
    }
  }, [open, isEdit, editStrategyId])

  // 未手动命名时自动生成: 前置 + 触发
  useEffect(() => {
    if (nameDirty) return
    const pre = available.find(s => s.id === preId)?.name ?? preId
    const trigger = available.find(s => s.id === triggerId)?.name ?? triggerId
    if (!pre && !trigger) {
      setName('')
      return
    }
    setName([pre, trigger].filter(Boolean).join(' → '))
  }, [preId, triggerId, available, nameDirty])

  const optionsFor = (role: 'pre' | 'trigger') =>
    available.filter(s => s.id !== (role === 'pre' ? triggerId : preId))

  const handleSave = async () => {
    if (!preId || !triggerId) {
      setError('请选择前置策略与触发策略')
      return
    }
    if (preId === triggerId) {
      setError('前置策略与触发策略不能相同')
      return
    }
    setSaving(true)
    setError('')
    try {
      const res = await api.strategySaveSequential({
        strategy_id: strategyId,
        name: name || strategyId,
        description,
        pre_strategy_id: preId,
        trigger_strategy_id: triggerId,
        max_gap_days: maxGap,
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

  const renderSelect = (role: 'pre' | 'trigger', value: string, onChange: (v: string) => void) => (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="w-full rounded-btn border border-border bg-elevated px-2 py-1.5 text-xs text-foreground"
    >
      <option value="">请选择策略…</option>
      {optionsFor(role).map((s: ScreenerStrategy) => (
        <option key={s.id} value={s.id}>
          {s.name}（{s.id}）
        </option>
      ))}
    </select>
  )

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
                先满足前置策略，再在 N 个交易日内满足触发策略
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
                    placeholder="先放量突破 → 再回踩确认"
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

              <div className="space-y-2 rounded-btn border border-border bg-elevated/40 p-3">
                <div className="flex items-center gap-2 text-[11px] text-muted">
                  <span className="rounded bg-cyan-500/15 px-1.5 py-0.5 font-medium text-cyan-300">1 前置</span>
                  先满足的策略（决定候选股票）
                </div>
                {renderSelect('pre', preId, setPreId)}

                <div className="flex items-center gap-2 pt-1 text-[11px] text-muted">
                  <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-medium text-emerald-300">2 触发</span>
                  之后满足的策略（决定入选日与评分）
                </div>
                {renderSelect('trigger', triggerId, setTriggerId)}

                <div className="flex items-center gap-2 pt-1">
                  <label className="text-[11px] text-muted">最大间隔</label>
                  <input
                    type="number"
                    min={0}
                    max={MAX_GAP}
                    value={maxGap}
                    onChange={e =>
                      setMaxGap(Math.max(0, Math.min(MAX_GAP, parseInt(e.target.value) || 0)))
                    }
                    className="w-20 rounded-btn border border-border bg-base px-2 py-1 text-xs text-foreground"
                  />
                  <span className="text-[11px] text-muted">
                    个交易日（0 = 只认同日命中；当日命中记为间隔 0）
                  </span>
                </div>
              </div>

              <div className="rounded-btn border border-border/60 bg-elevated/30 px-3 py-2 text-[11px] leading-relaxed text-muted">
                判定规则：触发策略命中的当天，若前置策略在「当日或之前 N 个交易日内」命中过，则入选；
                评分取触发策略当日评分，并附带间隔天数。中间出现卖点不会作废。
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
