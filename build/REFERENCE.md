## 8. Library API Reference (Verified Patterns)

When in doubt about an API, verify with these exact snippets before using them elsewhere.

### Playwright — async

```python
from playwright.async_api import async_playwright

async def run():
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto("http://localhost:9000/page/x")
    snapshot = await page.accessibility.snapshot()  # returns dict or None
    await browser.close()
    await pw.stop()
```

Accessibility snapshot node shape (simplified): `{"role": str, "name": str, "value": str, "children": [...], "selected": bool?, "checked": bool?}`.

### FastAPI — adding routes alongside create_app

```python
from fastapi.responses import HTMLResponse

app = create_app(...)

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<html>...</html>"
```

### Pydantic v2

- Use `model_dump()` not `.dict()`
- Use `model_dump_json()` not `.json()`
- Use `Field(default=...)` not `Field(default_value=...)`

### Jinja2

- Autoescape enabled → use `{{ var }}` for text, `{% raw %}...{% endraw %}` for literal JS
- `| lower`, `| replace` filters available
- `{% include '_base_styles.jinja' %}` for partials

### OpenEnv

- `create_app(EnvClass, ActionModel, ObservationModel, env_name=..., max_concurrent_envs=1)` returns a FastAPI app
- `Environment` subclass must implement `reset`, `step`, `state` (property)

---

