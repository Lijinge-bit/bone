"use strict";

const ICONS = {
  dashboard:'<rect x="3" y="3" width="7" height="7" rx="1.4"/><rect x="14" y="3" width="7" height="7" rx="1.4"/><rect x="3" y="14" width="7" height="7" rx="1.4"/><rect x="14" y="14" width="7" height="7" rx="1.4"/>',
  folder:'<path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><path d="M3 9h18"/>',
  chart:'<path d="M4 3v17h17M8 15l4-5 4 3 5-7"/>',
  book:'<path d="M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1v15Z"/>',
  shield:'<path d="m12 3 8 3v6c0 5-8 9-8 9S4 17 4 12V6Z"/><path d="m8 11 3 3 5-5"/>',
  upload:'<path d="M12 16V3m-5 5 5-5 5 5M4 16v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/>',
  plus:'<path d="M12 4v16M4 12h16"/>',
  arrow:'<path d="M4 12h16m-6-6 6 6-6 6"/>',
  back:'<path d="M20 12H4m6-6-6 6 6 6"/>',
  search:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
  close:'<path d="m6 6 12 12M6 18 18 6"/>',
  check:'<path d="m5 12 4 4L19 6"/>',
  checkcircle:'<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  clock:'<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  scan:'<path d="M8 3H4a1 1 0 0 0-1 1v4m13-5h4a1 1 0 0 1 1 1v4M3 16v4a1 1 0 0 0 1 1h4m8 0h4a1 1 0 0 0 1-1v-4M7 12h10"/>',
  layers:'<path d="m12 3 10 6-10 6L2 9Zm-9 11 9 5 9-5M3 18l9 5 9-5"/>',
  info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.1"/>',
  warning:'<path d="m12 3 10 18H2Z"/><path d="M12 9v5m0 3v.1"/>',
  zoomin:'<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6m-3-3v6"/>',
  zoomout:'<circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6"/>',
  reset:'<path d="M3 10a9 9 0 1 1 2 8M3 4v6h6"/>',
  eye:'<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
  print:'<path d="M6 9V3h12v6M6 17H3v-7h18v7h-3M6 14h12v7H6Zm11-3h1"/>',
  image:'<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8" cy="8" r="2"/><path d="m3 17 5-5 4 4 4-6 5 7"/>',
};
const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ICONS.info}</svg>`;
const $ = (s, root=document) => root.querySelector(s);
const $$ = (s, root=document) => [...root.querySelectorAll(s)];
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = value => value == null ? '—' : Number(value).toLocaleString('zh-CN');
const month = value => value == null ? '—' : Number(value).toFixed(1);
function date(value, full=false) {
  if (!value) return '—';
  return new Intl.DateTimeFormat('zh-CN', {timeZone:'Asia/Shanghai', ...(full?{year:'numeric'}:{}),month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(value));
}
function hydrateIcons(root=document) { $$('[data-icon]',root).forEach(el=>{el.innerHTML=icon(el.dataset.icon);}); }
function badge(label, color='') { return `<span class="badge ${color}"><i></i>${escapeHtml(label)}</span>`; }
function caseBadge(c) {
  if(c.qc_status==='rejected') return badge('已排除','red');
  if(c.qc_status==='pending') return badge('待质量复核','amber');
  const e=c.latest;
  if(e && !e.stale){
    if(['queued','running'].includes(e.status)) return badge('评估中','blue');
    if(e.status==='failed') return badge('评估失败','red');
    if(e.reviews?.length) return badge('已人工复核','green');
    if(['success','partial'].includes(e.status)) return badge('待结果复核','blue');
  }
  return badge(e?.stale?'输入已更新':'待评估');
}
const state={status:null,report:null,cases:[],case:null,total:0,filter:'all',search:'',offset:0,pageSize:12,route:'overview',navEpoch:0,selectedFile:null,previewURL:null,view:{zoom:1,x:0,y:0,brightness:100,contrast:100,heat:false,opacity:65}};
let pollTimer=null,searchTimer=null;

async function api(path,options={}) {
  const response=await fetch(path,{...options,headers:{...(options.body instanceof FormData?{}:{'Content-Type':'application/json'}),...options.headers}});
  let data;
  try {data=await response.json();} catch {throw new Error('服务返回格式异常，请检查连接');}
  if(!response.ok){
    const detail=data.detail;
    const message=typeof detail==='string'?detail:Array.isArray(detail)?'输入格式有误，请检查必填字段':detail?.message||'操作失败，请重试';
    const error=new Error(message);error.status=response.status;error.detail=detail;throw error;
  }
  return data;
}
function toast(message,error=false){const el=document.createElement('div');el.className=`toast ${error?'error':''}`;el.textContent=message;$('#toasts').append(el);setTimeout(()=>el.remove(),5000);}
function loading(){return '<div class="page-loading"><span class="spinner"></span>正在载入…</div>';}
function empty(title,description,button=true){return `<div class="empty"><div class="empty-icon">${icon('image')}</div><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p>${button?`<button class="button primary" data-action="upload">${icon('plus')}导入第一张影像</button>`:''}</div>`;}
function heading(title,desc,tag='WORKSPACE',action=true){return `<div class="page-heading"><div><span class="eyebrow">${tag}</span><h1>${title}</h1><p>${desc}</p></div>${action?`<button class="button primary" data-action="upload">${icon('plus')}导入影像</button>`:''}</div>`;}
async function refreshStatus(){state.status=await api('/api/status');$('#nav-count').textContent=state.status.stats.total;$('#connection').className='connection';$('#connection').innerHTML='<i></i>服务已连接';}

function metrics(){const s=state.status.stats;return `<div class="metrics">${[
  ['样本总数',s.total,'folder','当前研究空间的影像记录',''],
  ['待质量复核',s.pending,'clock','完成复核后可开始评估','amber'],
  ['已完成评估',s.assessed,'scan','与当前输入版本一致的结果',''],
  ['已人工复核',s.reviewed,'checkcircle','保留模型原值与复核意见',''],
].map(([label,value,symbol,note,color])=>`<article class="metric"><div class="metric-top"><span class="metric-label">${label}</span><span class="metric-icon ${color}">${icon(symbol)}</span></div><div class="metric-number">${num(value)}</div><div class="metric-note"><span class="dot"></span>${note}</div></article>`).join('')}</div>`;}
function sampleTable(cases){return `<div class="table-wrap"><table><thead><tr><th>样本 / 影像</th><th>性别</th><th>处理状态</th><th>预测骨龄</th><th>导入时间</th><th></th></tr></thead><tbody>${cases.map(c=>{
  const e=c.latest;const prediction=e&&!e.stale&&['success','partial'].includes(e.status)?`${month(e.prediction_months)} <small>月</small>`:'—';
  return `<tr><td><a class="sample-cell" href="#case/${c.id}"><img class="sample-thumb" src="${c.image_url}" alt="手部影像缩略图" loading="lazy"><div><span class="sample-name">${escapeHtml(c.source_image_id?`RSNA · ${c.source_image_id}`:c.filename)}</span><small>${c.width} × ${c.height} · PNG</small></div></a></td><td>${c.sex==='M'?'男':c.sex==='F'?'女':'待确认'}</td><td>${caseBadge(c)}</td><td>${prediction}</td><td>${date(c.created_at)}</td><td><a class="text-link" href="#case/${c.id}">打开 ${icon('arrow')}</a></td></tr>`;
}).join('')}</tbody></table></div>`;}
function modelCard(){const m=state.status.model;const e=m?.evaluation;const mae=e?.test?.mae_months;return `<section class="panel model-card"><div class="model-card-top"><span class="model-icon">${icon('layers')}</span>${badge(m?'研究基线可用':'模型准备中',m?'green':'amber')}</div><h3>${m?'CPU 骨龄回归基线':'等待模型注册'}</h3><p>${m?'影像与性别共同输入，原图坐标关注区域解释。':'训练、独立评估并注册模型后，预测功能将自动开放。'}</p><div class="model-details"><div class="detail-row"><span>推理环境</span><strong>CPU · 单时相</strong></div><div class="detail-row"><span>留出测试 MAE</span><strong>${mae==null?'尚未评估':`${month(mae)} 个月`}</strong></div><div class="detail-row"><span>目标 &lt; 6 个月</span><strong>${!e?'尚未验证':e.target_passed?'已达到':'尚未达到'}</strong></div></div><div class="model-alert">${m?'这是公开数据上的研究基线，患者级隔离尚未验证，不用于临床诊断。':'工作台可以先导入影像并完成质量复核。系统不会生成虚构的预测结果。'}</div><a class="button text" href="#data">查看模型与评估 ${icon('arrow')}</a></section>`;}
function sourceCard(){const a=state.status.dataset_audit;const n=a?.matched_training;const complete=a&&n===12611&&a.matched_unlabeled===200;return `<section class="panel source-card"><h3>研究数据来源</h3><div class="source-label">${icon('folder')}RSNA Bone Age · Kaggle</div><div class="detail-row"><span>有标签记录</span><strong>12,611</strong></div><div class="detail-row"><span>无标签记录</span><strong>200</strong></div><div class="source-progress"><span style="width:${a?Math.min(100,n/12611*100):0}%"></span></div><div class="detail-row"><span>图片审计</span><strong>${complete?(state.status.dataset_archive_available?'已完成全量检查':'附带原环境审计报告'):a?`${num(n)} 张已匹配`:'正在准备'}</strong></div><p>无标签图片仅用于预测展示，不参与模型误差计算。</p></section>`;}
function renderOverview(){return `${heading('骨龄评估工作台','从影像导入到人工复核，记录每一次判断。','RESEARCH WORKBENCH')}${metrics()}<div class="dashboard-grid"><div><section class="panel"><div class="panel-header"><div><h2>最近导入的样本</h2><p>保留原始影像，追溯每一轮评估</p></div><a href="#cases" class="button text">全部样本 ${icon('arrow')}</a></div>${state.cases.length?sampleTable(state.cases.slice(0,5)):empty('开始你的第一轮评估','导入一张手部 PNG，确认性别并完成图片质量复核。')}<div class="table-footer"><span>${num(state.status.stats.total)} 份影像记录 · 独立样本编号不代表患者身份</span><a href="#guide">了解工作流程</a></div></section><section class="panel workflow-panel"><div class="panel-header"><h2>一条可追溯的评估路径</h2><span class="badge">4 STEPS</span></div><div class="workflow">${[['导入影像','原始 PNG 与性别'],['检查质量','核对完整手部与干扰'],['骨龄评估','模型结果与热力图'],['人工复核','保存意见与结果版本']].map(([t,d],i)=>`<div class="workflow-item"><span class="step-number">0${i+1}</span><div><h3>${t}</h3><p>${d}</p></div></div>`).join('')}</div></section></div><div class="dashboard-aside">${modelCard()}${sourceCard()}</div></div>`;}

function renderCases(){return `${heading('样本工作台','管理研究影像、质量状态与历次评估记录。','SAMPLE LIBRARY')}<section class="panel"><div class="toolbar"><label class="search">${icon('search')}<input id="case-search" type="search" value="${escapeHtml(state.search)}" placeholder="搜索样本编号或文件名" aria-label="搜索样本"></label><div class="filters" aria-label="状态筛选">${[['all','全部'],['pending','待质检'],['ready','可评估'],['assessed','已评估'],['reviewed','已复核'],['rejected','已排除']].map(([key,label])=>`<button class="filter ${state.filter===key?'active':''}" data-action="filter" data-filter="${key}" aria-pressed="${state.filter===key}">${label}</button>`).join('')}</div></div>${state.cases.length?sampleTable(state.cases):empty(state.search||state.filter!=='all'?'没有符合条件的样本':'样本库还是空的',state.search||state.filter!=='all'?'试试其他关键词或处理状态。':'从一张 PNG 影像开始建立评估记录。',!state.search&&state.filter==='all')}<div class="table-footer"><span>共 ${num(state.total)} 条${state.total?` · 显示 ${state.offset+1}–${Math.min(state.offset+state.pageSize,state.total)}`:''}</span><div class="pager"><button class="button secondary" data-action="prev-page" ${state.offset===0?'disabled':''}>上一页</button><button class="button secondary" data-action="next-page" ${state.offset+state.pageSize>=state.total?'disabled':''}>下一页</button></div></div></section>`;}

function renderQuality(c){return `<section class="panel"><div class="panel-header"><h2>图片质量复核</h2>${badge(c.qc_status==='ready'?'已确认可用':c.qc_status==='rejected'?'已排除':'待人工复核',c.qc_status==='ready'?'green':c.qc_status==='rejected'?'red':'amber')}</div><div class="panel-body"><p class="qc-intro">请确认目标手部完整，检查额外手指、文字与裁剪。亮度提示仅作参考，不能代替人工判断。</p>${c.qc_flags.length?`<ul class="qc-flags">${c.qc_flags.map(f=>`<li>${icon('warning')}${escapeHtml(f.label)}</li>`).join('')}</ul>`:'<div class="inline-note">未触发自动亮度提示，仍需核对构图和解剖细节。</div>'}<form id="quality-form"><input class="qc-reviewer" name="reviewer" aria-label="质量复核人" required maxlength="80" placeholder="复核人姓名或编号" value="${escapeHtml(c.qc_reviewer||localStorage.getItem('bone_reviewer')||'')}"><textarea name="note" aria-label="质量复核说明" required minlength="2" rows="2" maxlength="2000" placeholder="说明目标手部、影像可用性或排除原因">${escapeHtml(c.qc_note)}</textarea><div class="quality-buttons"><button type="submit" class="button secondary" value="reject" name="decision">排除此图</button><button type="submit" class="button primary" value="usable" name="decision">${icon('check')}确认可用</button></div></form></div></section>`;}
function renderResult(c,e){const m=state.status.model;const active=e&&['queued','running'].includes(e.status);const valid=e&&['success','partial'].includes(e.status);const canRun=m&&c.sex&&c.qc_status==='ready'&&!active;return `<section class="panel result-card"><div class="panel-header"><h2>AI 辅助评估</h2>${badge(active?'处理中':valid?'研究结果':e?.status==='failed'?'运行失败':'尚未评估',active?'blue':valid?'green':e?.status==='failed'?'red':'')}</div><div class="panel-body">${e?.stale?'<div class="stale-note">输入已修改，以下为历史结果。请重新评估后复核。</div>':''}${valid?`<div class="prediction-value">${month(e.prediction_months)}<span>个月</span></div><p class="prediction-years">约 ${Math.floor(e.prediction_months/12)} 岁 ${Math.round(e.prediction_months%12)} 个月 · 单时相骨龄</p>`:active?'<div class="page-loading" style="padding:18px 0"><span class="spinner"></span>正在进行本轮评估…</div>':'<div class="prediction-value">—<span>个月</span></div>'}${e?.error?`<p class="result-note">${escapeHtml(e.error)}</p>`:''}<p class="result-note">${!m?'模型尚未注册。当前可先完成影像导入与质量复核。':!c.sex?'请先确认性别。':c.qc_status!=='ready'?'请先完成图片质量复核并确认可用。':'公开数据研究模型，非临床结论。未提供实足年龄，不做超前或落后判断。'}</p><button class="button primary full" data-action="assess" ${canRun?'':'disabled'}>${active?'<span class="spinner"></span>':icon('scan')}${valid?'重新评估骨龄':'开始骨龄评估'}</button>${valid&&!e.stale?`<button class="button secondary full" style="margin-top:9px" data-action="review">${icon('checkcircle')}记录复核意见</button>`:''}<p class="model-mini">${e?`本轮模型 ${escapeHtml(e.model_id)}<br>解释：${e.explanation_status==='available'?'相对关注热力图可用':e.explanation_status==='pending'?'等待生成':'不可用'}`:m?`可用模型 ${escapeHtml(m.model_id)}`:'系统不会提供固定或随机的演示预测'}</p></div></section>`;}
function renderCase(c){const e=c.evaluations[0];const heat=e?.heatmap_url;const reviews=c.evaluations.flatMap(ev=>ev.reviews.map(r=>({...r,model_id:ev.model_id,stale:ev.stale})));return `<div class="print-only">研究结果 · 非正式临床报告 · AI 结果需人工复核</div><div class="page-heading workspace-heading"><div><a href="#cases" class="back-link">${icon('back')}返回样本库</a><h1>${escapeHtml(c.source_image_id?`样本 ${c.source_image_id}`:c.filename)}</h1><div class="workspace-meta"><span>${c.width} × ${c.height} px</span><span>8 位灰度 PNG</span><span>导入于 ${date(c.created_at,true)}</span>${caseBadge({...c,latest:e})}</div></div><button class="button secondary" data-action="print">${icon('print')}打印研究记录</button></div><div class="workspace-layout"><section class="viewer"><div class="viewer-toolbar"><div><button class="icon-button" data-action="zoom-out" aria-label="缩小">${icon('zoomout')}</button><span id="zoom-value" class="viewer-tag">${Math.round(state.view.zoom*100)}%</span><button class="icon-button" data-action="zoom-in" aria-label="放大">${icon('zoomin')}</button><button class="icon-button" data-action="reset-view" aria-label="重置阅片显示">${icon('reset')}</button></div><div><span class="viewer-tag">ORIGINAL PNG</span><span class="viewer-tag">性别 ${c.sex||'未确认'}</span></div></div><div class="viewer-canvas" id="viewer-canvas"><div class="viewer-overlays"><span>RSNA · ${escapeHtml(c.source_image_id||c.id.slice(0,8))}</span><span>原始影像 / 仅供研究</span></div><div class="image-transform" id="image-transform"><img class="original-image" id="original-image" src="${c.image_url}" alt="当前样本原始手部 X 线影像" draggable="false">${heat?`<img id="heatmap-image" class="heatmap-image" src="${heat}" alt="本轮骨龄模型相对关注热力图" ${state.view.heat?'':'hidden'}>`:''}</div></div><div class="viewer-controls"><label>亮度<input id="brightness" type="range" min="40" max="220" value="${state.view.brightness}" aria-label="显示亮度"></label><label>对比度<input id="contrast" type="range" min="40" max="220" value="${state.view.contrast}" aria-label="显示对比度"></label><div class="heatmap-control"><label class="switch"><input id="heat-toggle" type="checkbox" ${state.view.heat&&heat?'checked':''} ${heat?'':'disabled'}>关注热力图</label><label class="heat-opacity">透明度<input id="heat-opacity" type="range" min="10" max="100" value="${state.view.opacity}" ${heat?'':'disabled'} aria-label="热力图透明度"></label></div></div><div class="viewer-foot"><span>拖动平移 · 滚轮缩放 · 调节显示不影响模型输入</span><span>${heat?'热力图表示相对关注，不是病变分割':'尚无本轮可用解释'}</span></div></section><aside class="case-side"><section class="panel"><div class="panel-header"><h2>评估输入</h2><span class="badge">版本 ${c.revision}</span></div><div class="panel-body"><div class="input-row"><label>性别 · ${c.sex_source==='csv'?'CSV 标签':'人工确认'}</label><div class="sex-options"><label><input type="radio" name="case-sex" value="M" ${c.sex==='M'?'checked':''}><span>男</span></label><label><input type="radio" name="case-sex" value="F" ${c.sex==='F'?'checked':''}><span>女</span></label></div></div><div class="detail-row"><span>实足年龄</span><strong>未提供 · 当前非必填</strong></div><div class="detail-row"><span>来源编号</span><strong>${escapeHtml(c.source_image_id||'未填写')}</strong></div></div></section>${renderQuality(c)}${renderResult(c,e)}</aside></div><div class="case-bottom"><section class="panel"><div class="panel-header"><h2>人工复核记录</h2><span class="badge">${reviews.length} 条</span></div>${reviews.length?`<div class="history-list">${reviews.map(r=>`<article class="history-item"><div class="history-top"><strong>${escapeHtml(r.reviewer)} · ${{accepted:'接受',question:'存疑',manual:'转人工处理'}[r.decision]}</strong><span>${date(r.created_at)}</span></div>${r.confirmed_months!=null?`<p>人工确认骨龄：${month(r.confirmed_months)} 个月</p>`:''}<p>${escapeHtml(r.note)}</p>${r.stale?'<p>基于历史输入版本</p>':''}</article>`).join('')}</div>`:'<div class="reviews-empty">完成骨龄评估后，可独立记录人工意见。AI 原始结果将始终保留。</div>'}</section><section class="panel"><div class="panel-header"><h2>历次评估</h2><span class="badge">${c.evaluations.length} 轮</span></div>${c.evaluations.length?`<div class="history-list">${c.evaluations.map(ev=>`<article class="history-item"><div class="history-top"><strong>${ev.prediction_months!=null?`${month(ev.prediction_months)} 个月`:'无有效结果'}</strong>${badge(ev.stale?'历史版本':({success:'已完成',partial:'解释不可用',queued:'排队中',running:'运行中',failed:'失败'}[ev.status]||ev.status),ev.stale?'amber':ev.status==='failed'?'red':'green')}</div><p>${date(ev.created_at)} · 输入版本 ${ev.input_revision} · ${ev.input_sex==='M'?'男':'女'}</p><p class="mono">${escapeHtml(ev.model_id)}</p></article>`).join('')}</div>`:'<div class="reviews-empty">尚未运行评估。每次输入或模型变更后重算，都会保留为独立版本。</div>'}</section></div>`;}

function renderData(){const a=state.report?.audit||state.status.dataset_audit;const e=state.report?.evaluation;const m=state.status.model;const test=e?.test||m?.evaluation?.test;return `${heading('数据与模型','将数据完整性、功能运行与模型效果分别验证。','DATA & EVALUATION',false)}<div class="notice-banner">${icon('info')}<span>图片编号不代表患者身份。当前数据未提供患者 ID 或检查日期，患者级隔离尚未验证；公开数据的研究结果不能直接外推至临床。</span></div><div class="data-grid"><div><section class="panel"><div class="panel-header"><h2>RSNA 图片审计</h2>${badge(a?'全量审计已生成':'数据准备中',a?'green':'amber')}</div><div class="panel-body"><div class="stat-pairs"><div class="stat-pair"><strong>${num(a?.matched_training)}</strong><span>已匹配的有标签图片 / 12,611</span></div><div class="stat-pair"><strong>${num(a?.matched_unlabeled)}</strong><span>已匹配的无标签图片 / 200</span></div><div class="stat-pair"><strong>${num(a?.anomalies_count)}</strong><span>缺失、冲突或解码异常</span></div><div class="stat-pair"><strong>${num(a?.exact_duplicate_groups)}</strong><span>相同像素内容的重复组</span></div></div><p class="report-note">${a?`完整 PNG 解码和 ZIP CRC 检查已执行。${state.status.dataset_archive_available?'本环境保留完整原图归档。':'这里展示随项目附带的原环境审计报告，本机未检测到完整原图归档。'}另有 ${num(a.dhash_candidate_groups)} 个相同差异哈希候选组；同组图片在划分时保持一致。近重复搜索并非穷尽，患者身份仍未知。`:'图片正在准备或尚未运行全量审计，当前不会以样例抽查代替完整匹配。'}</p></div></section><section class="panel"><div class="panel-header"><h2>独立数据划分</h2><span class="badge">固定清单 · 分组隔离</span></div><div class="panel-body">${a?`${Object.entries(a.splits).map(([key,value])=>`<div class="bar-row"><span>${{train:'训练',validation:'验证',test:'有标签留出测试'}[key]}</span><div class="bar-track"><div class="bar-fill" style="width:${value/a.matched_training*100}%"></div></div><strong>${num(value)}</strong></div>`).join('')}<p class="report-note">按性别与骨龄区间分层，并保持精确重复和同哈希候选组不跨分区。原始 200 张无标签图片单独保留，仅用于预测。</p>${state.status.unlabeled_predictions_available?'<a class="button secondary" href="/api/unlabeled-predictions" download>导出 200 张预测（含模型版本）</a>':''}<div class="detail-row"><span>划分种子</span><strong>${a.split_seed}</strong></div><div class="detail-row"><span>原图尺寸种类</span><strong>${num(a.distinct_dimensions)}</strong></div>`:empty('等待数据清单','完整审计后显示真实分区数量。',false)}</div></section><section class="panel"><div class="panel-header"><h2>骨龄分层误差</h2><span class="badge">留出测试 · 月</span></div>${test?`<div class="table-wrap"><table><thead><tr><th>参考骨龄范围</th><th>样本数</th><th>MAE（月）</th><th>说明</th></tr></thead><tbody>${Object.entries(test.by_boneage).map(([key,v])=>`<tr><td>${escapeHtml(key.replace('1000','+'))} 月</td><td>${num(v.n)}</td><td>${month(v.mae_months)}</td><td>${v.n<50?'样本较少，谨慎解读':'研究分层结果'}</td></tr>`).join('')}</tbody></table></div><div class="panel-body"><div class="detail-row"><span>男 / MAE</span><strong>${num(test.by_sex.M.n)} 例 / ${month(test.by_sex.M.mae_months)} 月</strong></div><div class="detail-row"><span>女 / MAE</span><strong>${num(test.by_sex.F.n)} 例 / ${month(test.by_sex.F.mae_months)} 月</strong></div></div>`:empty('还没有独立评估结果','模型训练与选择完成后，才运行固定留出测试集。',false)}</section></div><aside class="data-aside"><section class="panel"><div class="panel-header"><h2>模型效果</h2>${badge(test?'已评估':'未评估',test?'green':'amber')}</div><div class="panel-body"><div class="large-mae">${month(test?.mae_months)}<span>个月 MAE</span></div><div class="assessment-strip"><span class="muted">原文目标 &lt; 6 个月</span>${badge(test?(test.mae_months<6?'已达到':'尚未达到'):'尚未验证',test&&test.mae_months<6?'green':'amber')}</div><div class="detail-row"><span>测试样本数</span><strong>${num(test?.n)}</strong></div><div class="detail-row"><span>最佳轮次</span><strong>${num(e?.best_epoch||m?.evaluation?.best_epoch)}</strong></div><div class="detail-row"><span>选择依据</span><strong>验证集 MAE</strong></div><div class="detail-row"><span>输入处理</span><strong>${m?.image_size?`${m.image_size}×${m.image_size} 等比填充`:'待确认'}</strong></div><p class="report-note">CPU 轻量 CNN 从头训练基线，不是预训练 ResNet 或临床模型。模型目标未达时仍如实展示结果，后续可对照预训练模型改进。</p></div></section><section class="panel source-card"><h3>后续阶段的数据前提</h3><p>ROI 检测：骨骼类别和位置框标注。<br><br>DICOM：医院编码样本、元数据和脱敏流程。<br><br>联合预测：同一患者随访、检查日期、身高及成年身高与闭合结局。</p><a class="button text" href="/api/prd" target="_blank" rel="noopener">查看 PRD V0.2 ${icon('arrow')}</a></section></aside></div>`;}
function renderGuide(){return `${heading('从一张影像开始','了解输入要求、复核流程与研究边界。','GETTING STARTED',false)}<div class="guide-grid">${[
  ['01','准备图片与性别','使用原始 8 位灰度 PNG，单张不超过 15 MiB。确认性别为男或女；暂不清楚可以先导入，评估前再确认。实足年龄不是当前模型必填项。'],
  ['02','完成图片质量复核','查看完整手部，检查指尖、腕部、额外手指、文字与裁剪。亮度提示只是辅助，无法自动判定临床质量。填写复核人和原因，确认可用或排除。'],
  ['03','查看骨龄与模型关注','模型注册后可运行真实评估。单位为月，热力图表示相对关注区域，不是生长板分割、病变定位或模型置信度。显示调节不会改变模型输入。'],
  ['04','独立记录人工意见','接受、存疑或转人工处理，并记录依据。人工确认值与模型原值分开保留。输入更正后旧结果成为历史版本，需要重新评估。'],
].map(([n,t,d])=>`<section class="panel guide-card"><span class="guide-number">${n}</span><h2>${t}</h2><p>${d}</p></section>`).join('')}<section class="panel guide-card wide"><h2>当前能力与使用边界</h2><ul><li>首版支持单张影像骨龄评估，不生成成年身高、剩余生长高度、闭合时间窗或随访曲线。</li><li>模型在公开 RSNA 数据上训练与评估，患者级隔离和中国临床人群适用性未验证。</li><li>数据许可、参考标签标准和医院授权需按实际用途核实。当前服务面向本机单用户研究，远程和多人部署前需加入身份验证与访问控制。</li><li>系统尚未经过医院临床试用或认证，研究复核不等同临床报告签署。</li></ul><a href="/api/prd" class="button secondary" target="_blank" rel="noopener">${icon('book')}阅读完整 PRD V0.2</a></section></div>`;}

async function route({quiet=false}={}){
  const epoch=++state.navEpoch;
  clearTimeout(pollTimer);
  const raw=location.hash.slice(1)||'overview';
  const [name,id]=raw.split('/');
  const routeName=['overview','cases','data','guide','case'].includes(name)?name:'overview';
  state.route=routeName;
  if(!quiet)$('#main').innerHTML=loading();
  $$('.sidebar nav a').forEach(a=>a.classList.toggle('active',a.dataset.route===(routeName==='case'?'cases':routeName)));
  $('#breadcrumb-title').textContent={overview:'工作概览',cases:'样本工作台',case:'影像阅览与评估',data:'数据与模型',guide:'使用指南'}[routeName];
  try{
    await refreshStatus();
    let html;
    if(routeName==='case'){
      const c=await api(`/api/cases/${encodeURIComponent(id||'')}`);
      if(epoch!==state.navEpoch)return;
      if(state.case?.id!==c.id)state.view={zoom:1,x:0,y:0,brightness:100,contrast:100,heat:false,opacity:65};
      state.case=c;html=renderCase(c);
      if(c.evaluations.some(e=>['queued','running'].includes(e.status)))pollTimer=setTimeout(()=>route({quiet:true}),1300);
    }else if(routeName==='overview'||routeName==='cases'){
      const params=new URLSearchParams(routeName==='overview'?{limit:5}:{q:state.search,state:state.filter,limit:state.pageSize,offset:state.offset});
      const result=await api(`/api/cases?${params}`);if(epoch!==state.navEpoch)return;
      state.cases=result.items;state.total=result.total;
      html=routeName==='overview'?renderOverview():renderCases();
    }else if(routeName==='data'){
      state.report=await api('/api/dataset-report');if(epoch!==state.navEpoch)return;html=renderData();
    }else html=renderGuide();
    if(epoch!==state.navEpoch)return;
    $('#main').innerHTML=html;hydrateIcons($('#main'));
    if(routeName==='case'){updateViewer();bindViewer();}
  }catch(error){if(epoch!==state.navEpoch)return;$('#connection').className='connection error';$('#connection').innerHTML='<i></i>连接或数据异常';$('#main').innerHTML=`<div class="error-state"><div class="empty-icon">${icon('warning')}</div><h2>暂时无法打开工作空间</h2><p>${escapeHtml(error.message)}</p><button class="button primary" data-action="retry">重新加载</button></div>`;}
}

function updateViewer(){const v=state.view;const transform=$('#image-transform');if(!transform)return;transform.style.transform=`translate(${v.x}px,${v.y}px) scale(${v.zoom})`;$('#original-image').style.filter=`brightness(${v.brightness}%) contrast(${v.contrast}%)`;if($('#zoom-value'))$('#zoom-value').textContent=`${Math.round(v.zoom*100)}%`;const heat=$('#heatmap-image');if(heat){heat.hidden=!v.heat;heat.style.opacity=v.opacity/100;}}
function bindViewer(){const canvas=$('#viewer-canvas');let start=null;canvas.addEventListener('wheel',ev=>{ev.preventDefault();state.view.zoom=Math.max(.5,Math.min(4,state.view.zoom*(ev.deltaY<0?1.1:.9)));updateViewer();},{passive:false});canvas.addEventListener('pointerdown',ev=>{if(ev.button!==0)return;start={x:ev.clientX,y:ev.clientY,px:state.view.x,py:state.view.y};canvas.setPointerCapture(ev.pointerId);canvas.classList.add('dragging');});canvas.addEventListener('pointermove',ev=>{if(!start)return;state.view.x=start.px+ev.clientX-start.x;state.view.y=start.py+ev.clientY-start.y;updateViewer();});const stop=()=>{start=null;canvas.classList.remove('dragging');};canvas.addEventListener('pointerup',stop);canvas.addEventListener('pointercancel',stop);}
function showUpload(){const form=$('#upload-form');form.reset();state.selectedFile=null;if(state.previewURL)URL.revokeObjectURL(state.previewURL);state.previewURL=null;$('#upload-preview').hidden=true;$('#upload-preview').innerHTML='';$('#file-title').textContent='点击选择，或将影像拖到这里';$('#file-description').textContent='8 位灰度 PNG · 单张不超过 15 MiB';$('#upload-error').hidden=true;$('#upload-submit').disabled=false;$('#upload-dialog').showModal();}
function setFile(file){if(!file)return;const invalid=file.size>15*1024*1024?'单张图片不能超过 15 MiB':!file.name.toLowerCase().endsWith('.png')?'当前仅支持 PNG 图片':null;if(invalid){state.selectedFile=null;$('#upload-file').value='';$('#upload-preview').hidden=true;$('#file-title').textContent='请重新选择有效 PNG 图片';toast(invalid,true);return;}state.selectedFile=file;if(state.previewURL)URL.revokeObjectURL(state.previewURL);state.previewURL=URL.createObjectURL(file);$('#file-title').textContent=file.name;$('#file-description').textContent=`${(file.size/1024/1024).toFixed(2)} MiB · 将在服务端验证编码`;$('#upload-preview').hidden=false;$('#upload-preview').innerHTML=`<img src="${state.previewURL}" alt="待导入影像预览"><p>原图会独立保留。<br>导入后请核对性别与质量。</p>`;const input=$('[name=source_image_id]',$('#upload-form'));if(!input.value)input.value=file.name.replace(/\.png$/i,'');}

document.addEventListener('click',async ev=>{
  const button=ev.target.closest('[data-action]');if(!button||button.disabled)return;
  const action=button.dataset.action;
  if(action==='upload'){showUpload();return;}
  if(action==='close-upload'){$('#upload-dialog').close();return;}
  if(action==='close-review'){$('#review-dialog').close();return;}
  if(action==='retry'){route();return;}
  if(action==='filter'){state.filter=button.dataset.filter;state.offset=0;route({quiet:true});return;}
  if(action==='prev-page'||action==='next-page'){state.offset=Math.max(0,state.offset+(action==='next-page'?1:-1)*state.pageSize);route({quiet:true});return;}
  if(action==='zoom-in'||action==='zoom-out'){state.view.zoom=Math.max(.5,Math.min(4,state.view.zoom*(action==='zoom-in'?1.2:1/1.2)));updateViewer();return;}
  if(action==='reset-view'){state.view={zoom:1,x:0,y:0,brightness:100,contrast:100,heat:false,opacity:65};['brightness','contrast'].forEach(id=>{$('#'+id).value=100;});$('#heat-toggle').checked=false;$('#heat-opacity').value=65;updateViewer();return;}
  if(action==='print'){window.print();return;}
  if(action==='review'){const e=state.case?.evaluations[0];if(!e||e.stale)return;$('#review-form').reset();$('[name=reviewer]',$('#review-form')).value=localStorage.getItem('bone_reviewer')||'';$('#review-error').hidden=true;$('#review-dialog').showModal();return;}
  if(action==='assess'){
    button.disabled=true;
    try{await api(`/api/cases/${state.case.id}/assess`,{method:'POST',body:JSON.stringify({revision:state.case.revision})});toast('评估已提交，正在运行真实模型');await route({quiet:true});}
    catch(error){toast(error.message,true);button.disabled=false;}
  }
});
document.addEventListener('input',ev=>{
  const el=ev.target;
  if(el.id==='case-search'){state.search=el.value;state.offset=0;clearTimeout(searchTimer);const cursor=el.selectionStart;searchTimer=setTimeout(async()=>{await route({quiet:true});const input=$('#case-search');if(input){input.focus();input.setSelectionRange(cursor,cursor);}},300);}
  if(['brightness','contrast'].includes(el.id)){state.view[el.id]=Number(el.value);updateViewer();}
  if(el.id==='heat-opacity'){state.view.opacity=Number(el.value);updateViewer();}
});
document.addEventListener('change',async ev=>{
  if(ev.target.id==='upload-file')setFile(ev.target.files[0]);
  if(ev.target.id==='heat-toggle'){state.view.heat=ev.target.checked;updateViewer();}
  if(ev.target.name==='case-sex'){
    try{await api(`/api/cases/${state.case.id}/sex`,{method:'PATCH',body:JSON.stringify({sex:ev.target.value,revision:state.case.revision})});toast('性别已更新，旧结果保留为历史版本');route({quiet:true});}
    catch(error){toast(error.message,true);route({quiet:true});}
  }
});
document.addEventListener('submit',async ev=>{
  if(!['upload-form','quality-form','review-form'].includes(ev.target.id))return;
  ev.preventDefault();const form=ev.target;const values=Object.fromEntries(new FormData(form));
  if(form.id==='upload-form'){
    if(!state.selectedFile){$('#upload-error').textContent='请先选择一张 PNG 图片';$('#upload-error').hidden=false;return;}
    const submit=$('#upload-submit');submit.disabled=true;$('#upload-error').hidden=true;
    const data=new FormData();data.append('file',state.selectedFile);data.append('sex',values.sex||'');data.append('source_image_id',values.source_image_id||'');
    try{const c=await api('/api/cases',{method:'POST',body:data});$('#upload-dialog').close();location.hash=`case/${c.id}`;toast('影像已导入，请先进行质量复核');}
    catch(error){if(error.status===409&&error.detail?.case_id){$('#upload-dialog').close();location.hash=`case/${error.detail.case_id}`;toast(error.message);}else{$('#upload-error').textContent=error.message;$('#upload-error').hidden=false;}}
    finally{submit.disabled=false;}
  }else if(form.id==='quality-form'){
    const decision=ev.submitter?.value||'usable';const buttons=$$('button',form);buttons.forEach(b=>b.disabled=true);
    try{await api(`/api/cases/${state.case.id}/quality`,{method:'POST',body:JSON.stringify({...values,decision,revision:state.case.revision})});localStorage.setItem('bone_reviewer',values.reviewer);toast(decision==='usable'?'图片已确认可用':'图片已排除，原始文件仍保留');await route({quiet:true});}
    catch(error){toast(error.message,true);buttons.forEach(b=>b.disabled=false);}
  }else{
    const button=$('[type=submit]',form);button.disabled=true;$('#review-error').hidden=true;
    const payload={...values,confirmed_months:values.confirmed_months===''?null:Number(values.confirmed_months)};
    try{await api(`/api/evaluations/${state.case.evaluations[0].id}/reviews`,{method:'POST',body:JSON.stringify(payload)});localStorage.setItem('bone_reviewer',values.reviewer);$('#review-dialog').close();toast('复核意见已保存，模型原值保持独立');await route({quiet:true});}
    catch(error){$('#review-error').textContent=error.message;$('#review-error').hidden=false;}
    finally{button.disabled=false;}
  }
});
const drop=$('#drop-zone');
drop.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();$('#upload-file').click();}});
drop.addEventListener('dragover',ev=>{ev.preventDefault();drop.classList.add('dragging');});
drop.addEventListener('dragleave',()=>drop.classList.remove('dragging'));
drop.addEventListener('drop',ev=>{ev.preventDefault();drop.classList.remove('dragging');const files=ev.dataTransfer.files;if(files.length!==1){toast('请一次导入一张影像',true);return;}const transfer=new DataTransfer();transfer.items.add(files[0]);$('#upload-file').files=transfer.files;setFile(files[0]);});
window.addEventListener('hashchange',()=>{window.scrollTo({top:0});route();});
hydrateIcons();route();
