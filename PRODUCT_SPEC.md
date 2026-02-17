# Whiteboard Animator - Product Spec
*Competitor Analysis & Feature Roadmap*

## Executive Summary

Turn static images into whiteboard-style "hand drawing" animations. Unlike competitors using basic wipe reveals, we use AI stroke detection to trace actual drawing paths for realistic results.

---

## Competitor Landscape

### Doodly (Voomly Suite) - $49/month
**Target:** Marketers, YouTubers, course creators
**Model:** Desktop app (Mac/PC), subscription

**Features:**
- Drag-and-drop editor with timeline
- Multiple board styles: whiteboard, blackboard, glassboard, greenscreen
- "Smart Draw" - custom draw paths for imported images
- Large asset library: 200+ characters (20 poses each), scenes, props
- Multiple hand types (male/female, various ethnicities)
- Voiceover recording built-in
- Royalty-free music library
- Export 480p-1080p, 24-60fps
- People Builder (character customization)
- Talkia integration (AI voiceover)

**Weaknesses:**
- Desktop app only (no web)
- Complex UI - steep learning curve
- Expensive suite bundle
- Basic left-to-right drawing reveal

---

### VideoScribe (Sparkol) - £10-20/month (~$13-26)
**Target:** Enterprise, educators, HR teams
**Model:** Desktop app + web, subscription tiers

**Features:**
- AI image generation
- AI voiceover (45+ languages)
- AI script generation
- 5M+ premium images/icons
- Template library
- Multiple animation styles: draw, pulse, bounce
- Camera movements (zoom, pan)
- Various drawing hands, pens, erasers
- Reseller rights (Max tier)
- Custom fonts (Max tier)
- Team features

**Pricing tiers:**
- Lite: £10.83/mo - 5 downloads/mo, 5 min max, watermarks
- Core: £14.58/mo - 30 downloads/mo, 10 min max, no watermark
- Max: £20/mo - Unlimited, 20 min max, reseller rights

**Weaknesses:**
- Download limits on lower tiers
- Video length caps
- Watermarks unless paid
- AI features metered

---

### InstaDoodle - One-time ~$47-97
**Target:** Budget-conscious marketers
**Model:** Web-based, one-time payment (launch pricing)

**Features:**
- Simple upload → convert → download flow
- AI-powered conversion
- Quick turnaround

**Weaknesses:**
- Limited customization
- Basic output quality
- Typical launch-pricing/upsell funnel

---

### Renderforest - Freemium
**Target:** General video creation
**Model:** Web-based, freemium + subscription

**Features:**
- Whiteboard templates
- Part of larger video creation suite
- AI tools

**Weaknesses:**
- Whiteboard is just one small feature
- Template-based, not custom image conversion

---

## Our Differentiation

### Core Tech Advantage
**AI Stroke Tracing** - We analyze the actual strokes in an image and reveal them in natural drawing order. Circles draw as circles, text draws left-to-right per line, shapes follow their contours.

Competitors use:
- Simple left-to-right wipe
- Manual "draw path" creation (user has to trace each element)
- Template-based pre-drawn animations

We automate what Doodly's "Smart Draw" requires users to do manually.

---

## Feature Requirements

### MVP (Launch)

#### Core Functionality
- [x] **Upload any image** - JPG, PNG support ✅
- [x] **Auto sketch conversion** - Optional pencil/sketch style ✅
- [x] **AI stroke detection** - Analyze and trace actual drawing paths ✅
- [x] **Natural reveal animation** - Strokes appear as if hand-drawn ✅
- [x] **Export MP4** - 720p, 1080p options ✅
- [x] **Duration control** - 3-20 seconds adjustable ✅
- [x] **Speed/easing options** - Ease-out cubic ✅

#### Board Styles
- [x] **Whiteboard** - White background, dark strokes ✅
- [ ] **Blackboard** - Dark background, chalk effect
- [x] **Original colors** - Keep source image colors (--no-sketch) ✅

#### Web Interface
- [x] **Simple upload flow** - Drag & drop or file picker ✅
- [x] **Live preview** - See animation before export ✅
- [x] **Progress indicator** - Processing status ✅
- [x] **Download button** - Get MP4 ✅

### V1.1 (Post-Launch)

#### Hand Overlay (Optional)
- [ ] **Hand toggle** - On/off
- [ ] **Hand selection** - 6-8 diverse hand options
- [ ] **Hand position** - Follows stroke path

#### Audio
- [ ] **Background music** - 10-20 royalty-free tracks
- [ ] **Music volume control**
- [ ] **Custom audio upload**

#### Enhancements
- [ ] **Batch processing** - Multiple images at once
- [x] **Aspect ratios** - 16:9, 9:16 (vertical), 1:1 (square) ✅
- [ ] **Watermark** - Optional user logo overlay

### V2.0 (Growth)

#### AI Voiceover
- [ ] **Text-to-speech** - Generate voiceover from script
- [ ] **Multiple voices/languages**
- [ ] **Sync to animation** - Auto-time voiceover to video

#### Asset Library
- [ ] **Character library** - Pre-drawn people, icons
- [ ] **Scene templates** - Common layouts
- [ ] **Props/objects** - Business, education, general

#### Advanced Features
- [ ] **Timeline editor** - Control reveal order manually
- [ ] **Multi-scene videos** - Combine multiple images
- [ ] **Transitions** - Between scenes
- [ ] **Custom draw paths** - Manual override of AI detection

#### API
- [ ] **REST API** - Programmatic video generation
- [ ] **Webhooks** - Completion notifications
- [ ] **Bulk pricing** - Volume discounts

### V3.0 (Enterprise)

- [ ] **Team workspaces**
- [ ] **Brand kit** - Colors, fonts, logos
- [ ] **White-label** - Remove our branding
- [ ] **SSO/SAML** - Enterprise auth
- [ ] **Priority rendering**
- [ ] **SLA guarantees**

---

## Pricing Strategy

### Option A: Pay-Per-Video (Simpler)
- **Free tier:** 3 videos/month, 720p, watermarked
- **Basic:** $0.99/video - 1080p, no watermark
- **Pro:** $4.99/video - All features, priority rendering
- **Bulk packs:** 10 for $7, 50 for $30, 100 for $50

### Option B: Subscription (Recurring Revenue)
- **Free:** 3 videos/month, watermarked
- **Starter:** $9/month - 20 videos, 1080p
- **Pro:** $19/month - Unlimited videos, all features
- **Business:** $49/month - API access, team features

### Option C: Hybrid
- Free tier with watermark
- Pay-per-video for casual users
- Unlimited subscription for heavy users

**Recommendation:** Start with **Option A** (pay-per-video) - lower barrier to entry, easier to market, validates demand without commitment friction.

---

## Tech Stack (Suggested)

### Backend
- **Python** - Core video processing (OpenCV, ffmpeg)
- **FastAPI** - API server
- **Celery + Redis** - Job queue for async rendering
- **S3** - Video storage

### Frontend
- **Next.js** - React framework
- **Tailwind** - Styling
- **Uploadcare/Cloudinary** - Image upload handling

### Infrastructure
- **Vercel** - Frontend hosting
- **Railway/Render** - Backend + workers
- **Cloudflare R2** - Storage (cheaper than S3)

---

## Go-to-Market

### Target Audiences (Priority Order)
1. **YouTubers/Content Creators** - Explainer videos, tutorials
2. **Marketers** - Ads, social content
3. **Educators** - Lesson videos, course content
4. **Small Businesses** - Product demos, promos

### Channels
- Product Hunt launch
- YouTube tutorials/demos
- TikTok/Reels showing the tool
- SEO: "whiteboard animation maker", "doodle video creator"
- Appsumo deal (one-time purchase crowd)

### Positioning
> "Turn any image into a hand-drawn animation in 60 seconds. No templates. No timeline editing. Just upload and go."

Emphasize:
- **Speed** - Minutes, not hours
- **Simplicity** - No learning curve
- **Quality** - AI traces real strokes (show comparison videos)
- **Price** - Cheaper than subscriptions

---

## MVP Timeline (Estimated)

| Week | Milestone |
|------|-----------|
| 1 | Web UI skeleton, upload flow |
| 2 | Backend API, job queue |
| 3 | Integrate v3 animator, processing pipeline |
| 4 | Preview, download, polish |
| 5 | Payments (Stripe), user accounts |
| 6 | Testing, soft launch |

**MVP in 6 weeks** if focused.

---

## Open Questions

1. **Name?** - WhiteboardAI? DoodleForge? SketchMotion? InstaDraw?
2. **Domain?** - Check availability
3. **Initial pricing?** - Start low to get traction, raise later?
4. **Free tier limits?** - Too generous = no conversions, too stingy = no trials

---

## Next Steps

1. ✅ Core algorithm (v3 stroke tracing) - DONE
2. ✅ Build simple web interface - DONE
3. ✅ Set up backend API + job queue - DONE (FastAPI + background tasks)
4. [ ] Add custom domain (whiteboard.whinn.xyz)
5. [ ] Add Stripe payments
6. [ ] User accounts / auth
7. [ ] Soft launch, get feedback
8. [ ] Iterate based on user feedback
