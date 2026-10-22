"""Exercise the real UI and real registered model against the running local server.

Uses a public sample; review text explicitly identifies software verification.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import httpx
from playwright.sync_api import sync_playwright, expect


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--url',default='http://127.0.0.1:8000')
    ap.add_argument('--output',type=Path,default=Path(tempfile.gettempdir())/'bone-browser-check')
    ap.add_argument('--browser-executable',default=shutil.which('chromium') or shutil.which('google-chrome'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with httpx.Client(base_url=args.url,timeout=30) as client:
        cases=client.get('/api/cases').json()['items']
        sample=next(c for c in cases if c['source_image_id']=='1377')
        expected_mae=client.get('/api/status').json()['model']['evaluation']['test']['mae_months']
        sample_bytes=client.get(sample['image_url']).content
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.browser_executable,headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage','--no-zygote'],
            env={**os.environ,'XDG_CACHE_HOME':str(Path(tempfile.gettempdir())/'bone-browser-cache')})
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(args.url,wait_until='networkidle')
        expect(page.get_by_role('heading',name='骨龄评估工作台')).to_be_visible()
        page.get_by_role('button',name='导入影像',exact=True).click()
        page.locator('#upload-file').set_input_files({'name':'1377.png','mimeType':'image/png','buffer':sample_bytes})
        page.locator('#upload-form input[name=sex][value=F]').check()
        page.get_by_role('button',name='导入并检查').click()
        # Exact duplicate should open its original case, not create another record.
        expect(page.locator('#quality-form')).to_be_visible(timeout=15000)
        assert page.url.endswith('case/'+sample['id'])
        page.get_by_role('textbox',name='质量复核人').fill('自动化功能验收')
        page.get_by_role('textbox',name='质量复核说明').fill('公开样例管线验证：目标手部完整；本操作不代表临床质量审核。')
        page.get_by_role('button',name='确认可用',exact=True).click()
        run=page.get_by_role('button',name='开始骨龄评估',exact=True)
        if run.count()==0:run=page.get_by_role('button',name='重新评估骨龄',exact=True)
        expect(run).to_be_enabled(timeout=10000)
        run.click()
        expect(page.get_by_role('button',name='记录复核意见')).to_be_visible(timeout=60000)
        heat=page.locator('#heat-toggle')
        expect(heat).to_be_enabled()
        heat.check();expect(page.locator('#heatmap-image')).to_be_visible()
        page.locator('#brightness').fill('140')
        page.locator('#contrast').fill('110')
        page.get_by_role('button',name='放大',exact=True).click()
        page.get_by_role('button',name='记录复核意见').click()
        page.locator('#review-form input[name=reviewer]').fill('自动化功能验收')
        page.locator('#review-form select[name=decision]').select_option('question')
        review_note=f'功能验收 {time.time_ns()}：仅验证流程；模型精度尚未达到目标，不能作为临床判断。'
        page.locator('#review-form textarea[name=note]').fill(review_note)
        page.get_by_role('button',name='保存复核意见').click()
        expect(page.locator('#review-dialog')).not_to_be_visible(timeout=10000)
        expect(page.get_by_text(review_note,exact=True)).to_be_visible()
        page.screenshot(path=str(args.output/'evaluation.png'),full_page=True)
        page.get_by_role('link',name='样本工作台',exact=False).first.click()
        search=page.get_by_role('searchbox',name='搜索样本')
        search.fill('1377')
        expect(page.locator('tbody tr')).to_have_count(1,timeout=10000)
        page.get_by_role('link',name='数据与模型',exact=True).click()
        expect(page.get_by_role('heading',name='模型效果',exact=True)).to_be_visible()
        assert f'{expected_mae:.1f}' in page.locator('.large-mae').inner_text()
        page.screenshot(path=str(args.output/'data.png'),full_page=True)
        page.get_by_role('link',name='工作概览',exact=True).click()
        expect(page.get_by_role('heading',name='骨龄评估工作台')).to_be_visible()
        page.screenshot(path=str(args.output/'overview.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.screenshot(path=str(args.output/'mobile.png'),full_page=True)
        page.emulate_media(media='print')
        assert page.locator('.sidebar').evaluate('e=>getComputedStyle(e).display')=='none'
        browser.close()
    assert not errors, errors
    report={'status':'passed','browser_errors':errors,'checks':[
        'homepage','duplicate upload opens original','quality review','real model inference',
        'real heatmap','image controls','review persistence','search','data report','mobile width','print CSS']}
    (args.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
