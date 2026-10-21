// Reports a person's input to the automation process: trusted click, change and
// submit events, each with a description of the element taken now, because the
// element may be gone by the time the report is read (a button that closes its own
// dialog). Password values are never sent. Installed as an init script, so it runs
// in every document and frame, and once more by hand in the document already open.
(() => {
  if (window.__understudyRecorder) return;
  window.__understudyRecorder = true;
  // Looked up per report: the binding may be installed after this script runs.
  const send = (payload) => window.__understudyInput(payload);

  const norm = (s, n = 80) => (s || "").replace(/\s+/g, " ").trim().slice(0, n);
  const attr = (el, name) => (el.getAttribute ? el.getAttribute(name) : null);

  // A child-combinator path of nth-of-type steps from <html>: names exactly this
  // element in the page as it is now.
  const path = (el) => {
    const parts = [];
    for (let n = el; n && n !== document.documentElement; n = n.parentElement) {
      if (!n.parentElement) return null;
      const same = Array.from(n.parentElement.children).filter((c) => c.tagName === n.tagName);
      parts.unshift(`${n.tagName.toLowerCase()}:nth-of-type(${same.indexOf(n) + 1})`);
    }
    return "html > " + parts.join(" > ");
  };

  const describe = (el) => {
    const tag = el.tagName.toLowerCase();
    const type = (attr(el, "type") || "").toLowerCase() || null;
    const password = tag === "input" && type === "password";
    return {
      tag,
      type,
      id: el.id || null,
      name: attr(el, "name"),
      role: attr(el, "role"),
      aria_label: attr(el, "aria-label"),
      text: password ? "" : norm(el.innerText || el.textContent),
      value_attr: tag === "input" && !password && ["button", "submit", "reset", "image"].includes(type)
        ? attr(el, "value") : null,
      label: el.labels && el.labels.length ? norm(el.labels[0].innerText) : null,
      href: attr(el, "href"),
      autocomplete: attr(el, "autocomplete"),
      test_id: attr(el, "data-testid"),
      path: path(el),
    };
  };

  const report = (kind, el, extra) => {
    if (!el || !el.tagName) return;
    try {
      send({
        kind,
        ts: Date.now(),
        url: location.href,
        top: window === window.top,
        element: describe(el),
        ...extra,
      });
    } catch (e) {
      // A report that cannot be sent is lost; the page must not notice.
    }
  };

  const clickable = (el) =>
    el.closest("a, button, input, select, textarea, label, summary, [role=button], "
      + "[role=link], [role=checkbox], [role=radio], [onclick]") || el;

  window.addEventListener("click", (e) => {
    if (e.isTrusted && e.target instanceof Element) report("click", clickable(e.target), {});
  }, true);

  window.addEventListener("change", (e) => {
    const el = e.target;
    if (!e.isTrusted || !(el instanceof Element)) return;
    const extra = { value: null, option_label: null, checked: null };
    if (el.tagName === "SELECT") {
      const option = el.selectedOptions[0];
      extra.value = option ? option.value : "";
      extra.option_label = option ? norm(option.label || option.textContent) : "";
    } else if (el.type === "checkbox" || el.type === "radio") {
      extra.checked = el.checked;
    } else if (el.type !== "password") {
      extra.value = typeof el.value === "string" ? el.value.slice(0, 500) : null;
    }
    report("change", el, extra);
  }, true);

  window.addEventListener("submit", (e) => {
    if (e.isTrusted && e.target instanceof Element) report("submit", e.target, {});
  }, true);
})();
