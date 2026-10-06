import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

// ThreeUI embeds these CDN scripts in its iframe documents. Pin them at build
// time rather than editing the installed package; npm ci stays reproducible.
const instrumentSources: Record<string, true> = {
  'particle-network.html.js': true,
  'nexus-topology.html.js': true,
  'diagnostics-panel.html.js': true,
}
const cdnScripts = [
  {
    url: 'https://cdn.tailwindcss.com',
    pinnedUrl: 'https://cdn.tailwindcss.com/3.4.17',
    // This endpoint has no CORS header, so browsers cannot use SRI for it.
  },
  {
    url: 'https://code.iconify.design/iconify-icon/1.0.7/iconify-icon.min.js',
    integrity: 'sha384-ZBvlAMcOinSpqbKp+h0PpJxrDWCO8veRjvEhIc+Wg2Um8ZUKrbyNtJChA7FhNtCF',
  },
  {
    url: 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js',
    integrity: 'sha384-d+vyQ0dYcymoP8ndq2hW7FGC50nqGdXUEgoOUGxbbkAJwZqL7h+jKN0GGgn9hFDS',
  },
  {
    url: 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js',
    integrity: 'sha384-poC0r6usQOX2Ayt/VGA+t81H6V3iN9L+Irz9iO8o+s0X20tLpzc9DOOtnKxhaQSE',
  },
  {
    url: 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js',
    integrity: 'sha384-CI3ELBVUz9XQO+97x6nwMDPosPR5XvsxW2ua7N1Xeygeh1IxtgqtCkGfQY9WWdHu',
  },
]

function pinInstrumentScripts(): Plugin {
  return {
    name: 'pin-threeui-cdn-scripts',
    enforce: 'pre',
    transform(code, id) {
      if (!id.includes('/@designcodeio/threeui/lib-dist/shaders/neuform-isolated/sources/')
        || !instrumentSources[id.split('/').at(-1) ?? '']) return null

      for (const script of cdnScripts) {
        const integrity = script.integrity
          ? ` integrity="${script.integrity}" crossorigin="anonymous"`
          : ''
        code = code.replaceAll(
          `src="${script.url}"`,
          `src="${script.pinnedUrl ?? script.url}"${integrity}`,
        )
      }
      return { code, map: null }
    },
  }
}

export default defineConfig({
  base: '/verge-lab/',
  plugins: [pinInstrumentScripts(), react()],
})
