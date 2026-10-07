import assert from 'node:assert/strict'
import puppeteer from 'puppeteer'

// Run against an existing Vite server with Puppeteer available in the environment.
// Usage: node tools/check_demo.mjs http://localhost:5173/verge-lab/
const url = process.argv[2]
assert(url, 'Pass the demo URL as the first argument')

const checkDemo = async (page, isMobile) => {
  const results = []
  const nav = isMobile ? '.bottom-nav' : '.primary-nav'
  const heading = async (text) => {
    await page.waitForFunction(
      (title) => document.querySelector('h1')?.textContent === title, {}, text,
    )
  }
  const overflow = async (view) => {
    const dimensions = await page.evaluate(() => ({
      viewport: innerWidth,
      document: document.documentElement.scrollWidth,
      body: document.body.scrollWidth,
    }))
    assert(
      dimensions.document <= dimensions.viewport && dimensions.body <= dimensions.viewport,
      `${view} page overflow: ${JSON.stringify(dimensions)}`,
    )
    results.push({ view, ...dimensions })
  }
  const navigate = async (index, text) => {
    await page.click(`${nav} button:nth-child(${index})`)
    await heading(text)
    assert(await page.evaluate(() => document.activeElement.id === 'lab-main'), 'Navigation focus')
    assert(await page.evaluate(() => scrollY === 0), 'Navigation scroll reset')
    await overflow(text)
  }

  await page.keyboard.press('Tab')
  assert(await page.evaluate(() => document.activeElement.className === 'skip-link'),
    'Skip link first keyboard target')
  await page.keyboard.press('Enter')
  assert(await page.evaluate(() => document.activeElement.id === 'lab-main'),
    'Skip link focuses main')
  results.push('Skip link keyboard activation')
  await page.focus('.example-notice summary')
  await page.keyboard.press('Enter')
  assert(await page.$eval('.example-notice details', (element) => element.open),
    'Source disclosure opens by keyboard')
  await page.keyboard.press('Enter')
  await overflow('Overview')

  await navigate(2, 'Not every pair has a winner.')
  await page.focus('[aria-label="Zoom graph in"]')
  await page.keyboard.press('Enter')
  assert(await page.$eval('.graph-controls output', (element) => element.textContent === '125%'),
    'Keyboard zoom')
  await page.click('.graph-controls button:nth-child(2)')
  assert(await page.$eval('.graph-controls output', (element) => element.textContent === '100%'),
    'Reset zoom')
  const nodes = await page.$$eval('.graph-node', (elements) => elements.map((element) => element.id))
  await page.$eval(`#${nodes[0]}`, (element) => element.focus())
  await page.keyboard.press('ArrowRight')
  assert(await page.evaluate((id) => document.activeElement.id === id, nodes[1]),
    'Graph arrow key focus')
  await page.keyboard.press('Enter')
  assert(await page.$eval(`#${nodes[1]}`, (element) => element.getAttribute('aria-pressed') === 'true'),
    'Graph selection')
  await page.focus('.candidate-table th button')
  await page.keyboard.press('Enter')
  assert(await page.$eval('.candidate-table th button',
    (element) => element.getAttribute('aria-pressed') === 'true'), 'Table selection')
  await page.focus('.candidate-register-disclosure > summary')
  await page.keyboard.press('Enter')
  assert(await page.$eval('.candidate-register-disclosure', (element) => !element.open),
    'Table disclosure closes')
  await page.keyboard.press('Enter')

  const selects = await page.$$('.filter-bar select')
  const domain = await selects[0].$$eval('option', (elements) => elements[1].value)
  await selects[0].select(domain)
  assert(await page.$$eval('.candidate-table tbody td[data-label="Example domain"]',
    (elements, selected) => elements.every((element) => element.textContent === selected), domain),
    'Domain filter')
  await selects[0].select('all')
  await selects[1].select('ambiguous')
  assert(await page.$$eval('.graph-edge',
    (elements) => elements.every((element) => element.classList.contains('graph-edge--ambiguous'))),
    'Abstain graph filter')
  await selects[1].select('all')
  await page.focus('input[type="search"]')
  await page.type('input[type="search"]', 'zzzz-no-match')
  await page.waitForSelector('.empty-state')
  assert(await page.$eval('.filter-result strong', (element) => element.textContent === '0'),
    'Empty search count')
  await page.focus('input[type="search"]')
  await page.keyboard.down('Control')
  await page.keyboard.press('A')
  await page.keyboard.up('Control')
  await page.keyboard.press('Backspace')
  await page.waitForSelector('.graph-node')
  results.push('Graph zoom/reset, keyboard nodes/table, domain/verdict/search filters')

  await navigate(3, 'See why a pair was selected.')
  const pairSelects = await page.$$('.filter-bar select')
  await pairSelects[1].select('ambiguous')
  const chosen = await page.$eval('.pair-index button', (element) => element.dataset.pairId)
  await page.focus('.pair-index button')
  await page.keyboard.press('Enter')
  assert(await page.evaluate(() => document.activeElement.id === 'pair-evidence-title'),
    'Pair detail focus')
  assert(await page.$eval('.verdict-label', (element) => element.textContent.includes('Abstain')),
    'Abstention label')
  assert(await page.$$eval('.pair-choice header', (elements) => elements.every((element) =>
    !element.textContent.includes('Chosen') && !element.textContent.includes('Rejected'))),
  'No winner labels on abstention')
  await overflow('Pair abstention detail')
  if (isMobile) {
    await page.focus('.pair-back')
    await page.keyboard.press('Enter')
    assert(await page.evaluate((id) => document.activeElement.dataset.pairId === id, chosen),
      'Back restores pair focus')
  }
  await pairSelects[1].select('defended')
  await page.click('.pair-index button')
  assert(await page.$eval('.verdict-label', (element) => element.textContent.includes('Selected')),
    'Selected verdict')
  assert(await page.$$eval('.pair-choice header',
    (elements) => elements.some((element) => element.textContent.includes('Chosen'))),
  'Chosen answer label')
  await overflow('Selected pair detail')
  await pairSelects[1].select('all')
  results.push('Selected and abstained pair drilldown; no implied winner on abstention')

  await navigate(4, 'An authored example, not a run.')
  assert(await page.$eval('.checkpoint-trace svg', (element) => element.getAttribute('role') === 'img'),
    'Chart accessible description')
  await page.focus('.checkpoint-trace__plot')
  assert(await page.evaluate(() => document.activeElement.className === 'checkpoint-trace__plot'),
    'Chart scroll region focus')
  const mutation = await page.$('.mutation-register summary')
  if (mutation) {
    await mutation.focus()
    await page.keyboard.press('Enter')
    assert(await page.$eval('.mutation-register details', (element) => element.open),
      'Mutation disclosure keyboard')
  }
  await overflow('Artifact expanded mutation')
  results.push('Artifact chart focus, mutation disclosure')
  await navigate(1, 'Better on one score. Worse on none.')
  return { isMobile, results }
}

const browser = await puppeteer.launch({ headless: true })
try {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
    const page = await browser.newPage()
    const errors = []
    page.on('pageerror', (error) => errors.push(error.message))
    page.on('console', (message) => {
      if (['error', 'warn'].includes(message.type())) errors.push(message.text())
    })
    try {
      await page.setViewport(viewport)
      await page.goto(url, { waitUntil: 'networkidle0' })
      const result = await checkDemo(page, viewport.width < 768)
      assert.deepEqual(errors, [], 'Unexpected browser errors or console warnings')
      console.log(JSON.stringify({ viewport, ...result, errors }))
    } finally {
      await page.close()
    }
  }
} finally {
  await browser.close()
}
