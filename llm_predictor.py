"""
Gemini LLM predictor wrapper for football match outcomes.
- Tolerant numeric parsing (accept strings, % signs)
- Minimal JSON prompt (home_win, draw, away_win only)
- Smaller token budget by default
- Compatible with SDKs that don't support generate_content_async
"""
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List

import google.generativeai as genai
try:
    from google.generativeai.tool import Tool, GoogleSearchRetrieval
except Exception:
    Tool = None
    GoogleSearchRetrieval = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-pro")
class GeminiPredictor:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.2):
        self.model_id = model or DEFAULT_MODEL
        self.temperature = temperature
        self._client_ready = False
        # Base predictor model (strict minimal JSON)
        self.model = None
        # Advanced analysis model (richer JSON with summary and factors)
        self.model_advanced = None
        # Flash model for online-grounded context gathering
        self.model_flash = None

    async def initialize(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set; LLM predictor will be disabled")
            self._client_ready = False
            return
        try:
            # Enforce Gemini 2.5 Pro unless explicitly allowed otherwise
            desired = os.getenv("GEMINI_MODEL_ID") or "gemini-2.5-pro"
            allow_non_pro = os.getenv("ALLOW_NON_PRO") == "1"
            if not allow_non_pro:
                if ("2.5" not in desired) or ("pro" not in desired.lower()):
                    logger.warning("Overriding GEMINI_MODEL_ID to 'gemini-2.5-pro' (strict mode). Set ALLOW_NON_PRO=1 to override.")
                    desired = "gemini-2.5-pro"
            self.model_id = desired
            genai.configure(api_key=api_key)
            logger.info(f"Using Gemini model: {self.model_id}")
            # Base predictor model (strict minimal JSON)
            try:
                self.model = genai.GenerativeModel(
                    self.model_id,
                    system_instruction=(
                        "You are a concise probability JSON generator. "
                        "Always return only valid JSON with numeric fields: home_win, draw, away_win (0-100). "
                        "No prose, no code fences, no extra keys."
                    ),
                )
            except TypeError:
                self.model = genai.GenerativeModel(self.model_id)
            # Advanced analysis model (structured, brief rationale and factors)
            try:
                self.model_advanced = genai.GenerativeModel(
                    self.model_id,
                    system_instruction=(
                        "You are an expert football match analyst. Return ONLY valid JSON with this exact schema: "
                        "{home_win:number, draw:number, away_win:number, prediction:string, confidence:number, summary:string, factors:[{name:string, impact:number, evidence:string}]} . "
                        "Consider 50+ professional criteria used by sportsbooks and analytics firms (e.g., team strength, recent form (5-10), H2H, lineup quality, injuries/suspensions, fatigue, schedule density, rest days, travel, home advantage, venue effects, crowd intensity, tactics and formations, managerial styles, tactical matchups, xG/xGA, shot quality, PPDA, pressing intensity, build-up speed, set-pieces for/against, aerial duels, transitions, counter-pressing vulnerability, ball progression, turnovers, GK quality, finishing variance, penalties, discipline risk, referee tendencies, weather, pitch conditions, altitude, timezone, motivation, competition/stage/stakes, rotation risk, odds movement, market agreement, bookmaker margin, squad depth, bench impact, experience vs youth, captain influence, morale, contract/transfer noise, injuries returning, key absences, travel distance, short rest, etc.). "
                        "Output must be brief but cutting-edge: summary 5-8 lines max. Factors should list the top 8-12 most influential items only with a short evidence sentence each. Do not include code fences or extra keys. "
                        "Be current and avoid outdated claims. Do NOT fabricate injuries, lineups, or dates. Express uncertainty ONLY in the 'summary'; you MUST always set 'prediction' to one of: 'Home Win', 'Away Win', or 'Draw' (never 'uncertain')."
                    ),
                )
            except TypeError:
                # Fallback without system instruction
                self.model_advanced = genai.GenerativeModel(self.model_id)
            # Flash model (online-grounded research)
            try:
                flash_id = os.getenv("GEMINI_FLASH_MODEL_ID") or "gemini-2.5-flash"
                self.model_flash = genai.GenerativeModel(
                    flash_id,
                    system_instruction=(
                        "You are a web research agent with Google Search grounding. "
                        "Collect the latest, reliable football metrics for two teams. "
                        "Return ONLY valid JSON with schema: {report:string, sources:[{title:string,url:string}], timestamp:string}. "
                        "The 'report' should be concise but comprehensive, citing season context and recency. Do not fabricate."
                    ),
                )
            except TypeError:
                self.model_flash = genai.GenerativeModel(os.getenv("GEMINI_FLASH_MODEL_ID") or "gemini-2.5-flash")
            self._client_ready = True
            logger.info("Gemini predictor initialized (base + advanced + flash)")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini predictor: {e}")
            self._client_ready = False

    def _normalize_probs(self, home: float, draw: float, away: float) -> Dict[str, float]:
        # clamp and renormalize to 100
        vals = [max(0.0, min(100.0, float(v))) for v in (home, draw, away)]
        s = sum(vals) or 1.0
        return {
            "home_win": float(vals[0] / s * 100.0),
            "draw": float(vals[1] / s * 100.0),
            "away_win": float(vals[2] / s * 100.0),
        }

    def _coerce_prob(self, v, default: float) -> float:
        try:
            if v is None:
                return float(default)
            if isinstance(v, (int, float)):
                return float(v)
            # If string, strip percent and non-numeric chars conservatively
            if isinstance(v, str):
                s = v.strip().replace('%', '')
                # Extract first number pattern
                import re
                m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
                if m:
                    return float(m.group(0))
                return float(default)
            return float(default)
        except Exception:
            return float(default)

    def _extract_probs_from_text(self, s: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        import re
        def grab(keys):
            for key in keys:
                m = re.search(rf"{key}" + r"\"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", s, re.IGNORECASE)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        pass
            return None
        home = grab(['home_win','home'])
        draw = grab(['draw'])
        away = grab(['away_win','away'])
        return home, draw, away

    def _sanitize_json_string(self, s: str) -> str:
        import re
        s = s.strip().strip('`').replace('\n', ' ').replace('\r', ' ')
        # trim to last closing brace if needed
        if '{' in s and '}' not in s:
            s = s + '}'
        if '{' in s and '}' in s:
            s = s[:s.rfind('}')+1]
        # remove trailing commas before } or ]
        s = re.sub(r',\s*([}\]])', r'\1', s)
        s = re.sub(r'\s{2,}', ' ', s)
        return s

    def _extract_response_text(self, resp) -> Optional[str]:
        """Safely extract text from a Gemini response without triggering resp.text ValueError."""
        try:
            cands = getattr(resp, "candidates", []) or []
            for c in cands:
                content = getattr(c, "content", None)
                if not content:
                    continue
                parts = getattr(content, "parts", []) or []
                for p in parts:
                    if hasattr(p, "text") and getattr(p, "text", None):
                        return p.text
                    inline = getattr(p, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        import base64
                        try:
                            data_bytes = base64.b64decode(inline.data)
                            return data_bytes.decode('utf-8', errors='ignore')
                        except Exception:
                            continue
        except Exception:
            pass
        # Very last resort
        try:
            t = getattr(resp, "text", None)
            if isinstance(t, str) and t:
                return t
        except Exception:
            return None
        return None

    async def _generate(self, prompt: str, generation_config: Dict):
        """Call Gemini (base model) with async if available, otherwise sync in executor."""
        return await self._generate_with(self.model, prompt, generation_config)

    async def _generate_with(self, model, prompt: str, generation_config: Dict, tools: Optional[list] = None, tool_config: Optional[Dict] = None):
        """Call given Gemini model with async if available, otherwise sync in executor."""
        try:
            gen_async = getattr(model, 'generate_content_async', None)
            if callable(gen_async):
                if tools is not None or tool_config is not None:
                    return await gen_async(prompt, generation_config=generation_config, tools=tools, tool_config=tool_config)
                return await gen_async(prompt, generation_config=generation_config)
        except Exception:
            # Fall through to sync
            pass
        # Sync path in executor
        import asyncio
        loop = asyncio.get_running_loop()
        def _call():
            if tools is not None or tool_config is not None:
                return model.generate_content(prompt, generation_config=generation_config, tools=tools, tool_config=tool_config)
            return model.generate_content(prompt, generation_config=generation_config)
        return await loop.run_in_executor(None, _call)

    async def predict(self, home_team: str, away_team: str, date: Optional[str] = None) -> Dict:
        if not self._client_ready:
            return {"error": "LLM not configured"}

        # Minimal prompt; keep concise and schema-aligned
        prompt = (
            f"Matchup: {home_team} vs {away_team}. "
            "Return only valid JSON with numeric fields: home_win, draw, away_win (0-100). Be concise."
        )

        import asyncio, random, traceback
        data = None
        debug_info: Dict = {"stage": "pre-call"}
        debug_mode = os.getenv("DEBUG_LLM") == "1"
        # Seed debug info early so we have context even on early exceptions
        debug_info.update({
            "prompt_snippet": (f"{home_team} vs {away_team}")[:160],
            "temperature": self.temperature,
        })

        # Single-attempt design to avoid wasted retries
        attempt_specs = [
            {"prompt": prompt}, 
        ]

        for idx, spec in enumerate(attempt_specs):
            try:
                resp = await self._generate(
                    spec["prompt"],
                    generation_config={
                        "temperature": self.temperature if idx == 1 else 0.2,
                        "response_mime_type": "application/json",
                    },
                )
                # Extract diagnostics
                cands = getattr(resp, "candidates", []) or []
                finish_reasons = [getattr(c, "finish_reason", None) for c in cands]
                safety = [getattr(c, "safety_ratings", None) for c in cands]
                debug_info = {
                    "candidates_count": len(cands),
                    "finish_reasons": finish_reasons,
                    "safety": safety,
                }
                # Pull text payload (scan all parts)
                extracted = None
                if hasattr(resp, "text") and resp.text:
                    extracted = resp.text
                else:
                    for c in cands:
                        content = getattr(c, "content", None)
                        if not content:
                            continue
                        parts = getattr(content, "parts", []) or []
                        for p in parts:
                            if hasattr(p, "text") and p.text:
                                extracted = p.text
                                break
                            inline = getattr(p, "inline_data", None)
                            if inline and getattr(inline, "data", None):
                                import base64
                                try:
                                    data_bytes = base64.b64decode(inline.data)
                                    extracted = data_bytes.decode('utf-8', errors='ignore')
                                    break
                                except Exception:
                                    continue
                        if extracted:
                            break
                if extracted:
                    data = extracted
                    if debug_mode:
                        try:
                            snippet = str(extracted).replace('\n',' ')[:200]
                            logger.warning(f"Gemini raw payload (snippet): {snippet}")
                        except Exception:
                            pass
                    break  # success
                # Safety blocked - bail to fallback
                if any(ratings for ratings in safety):
                    logger.warning("Gemini response blocked by safety: %s", safety)
                    break
            except Exception as ex:
                logger.exception(f"Gemini call failed on attempt {idx+1}")
                continue

        if not data:
            msg = "Gemini returned no extractable data"
            if debug_mode:
                logger.warning(f"{msg}. Details: %s", debug_info)
            else:
                logger.warning(msg)
            return {"error": msg}

        # Parse JSON with sanitizer; tolerate string/percent values
        import json
        obj = None
        try:
            obj = json.loads(data)
        except Exception as e1:
            logger.warning(f"Failed to parse LLM JSON: {e1}; raw={str(data)[:200]}")
            data_clean = self._sanitize_json_string(data)
            try:
                obj = json.loads(data_clean)
            except Exception as e2:
                logger.warning(f"Sanitized JSON still invalid: {e2}; sanitized={data_clean[:200]}")
                # fallback to regex extraction; return raw string probabilities if found, else error
                h, d, a = self._extract_probs_from_text(data_clean)
                if h is None and d is None and a is None:
                    return {"error": "LLM JSON parse failed"}
                probs = self._normalize_probs(h, d, a)
                return {**probs, "framework": "gemini-llm", "source": "gemini-2.5", "raw": data_clean}

        # Coerce numbers even if JSON parsed with wrong types
        try:
            home_raw = obj.get("home_win", 33.3)
            draw_raw = obj.get("draw", 33.3)
            away_raw = obj.get("away_win", 33.4)
        except Exception:
            home_raw, draw_raw, away_raw = 33.3, 33.3, 33.4

        home = self._coerce_prob(home_raw, 33.3)
        draw = self._coerce_prob(draw_raw, 33.3)
        away = self._coerce_prob(away_raw, 33.4)

        # If invalid (NaN), try raw text fallback; otherwise do not invent heuristics
        if any(v != v for v in (home, draw, away)):  # NaN check
            h, d, a = self._extract_probs_from_text(self._sanitize_json_string(str(data)))
            if h is None or d is None or a is None:
                return {"error": "LLM numeric coercion failed", "raw": str(data)[:200]}
            home, draw, away = h, d, a

        probs = self._normalize_probs(home, draw, away)
        return {
            **probs,
            "framework": "gemini-llm",
            "source": self.model_id,
            "raw": data,
        }

    async def _gather_online_context(self, home_team: str, away_team: str, date: Optional[str] = None) -> Optional[Dict]:
        """
        Use Gemini 2.5 Flash with Google Search grounding to gather fresh context.
        Returns dict: { report: string, sources: [{title,url}], timestamp: string } or None on failure.
        """
        try:
            if not self.model_flash:
                return None
            # Compute strict recency window for dynamic stats (default last 120 days)
            now = datetime.now(timezone.utc)
            days_back = int(os.getenv("GROUNDING_DAYS_BACK", "120"))
            start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
            end_date = now.strftime("%Y-%m-%d")

            query_header = (
                f"Research the latest football metrics and news for: {home_team} vs {away_team}" + (f" on {date}." if date else ".")
            )
            requirements = (
                "Use at least 12 diverse, reputable sources (league/team sites, stats providers, injury trackers, pressers, recent match reports, major news). "
                "STRICT RECENCY for dynamic stats: Only include items dated between " + start_date + " and " + end_date + ". Ignore older items unless they are season-long official references. "
                "Prioritize current season context and last 5-10 matches. Verify current managers/coaches, injuries/suspensions, and lineup availability. "
                "Also gather HISTORICAL CONTEXT (beyond the window) strictly for rivalry/head-to-head trends: long-run H2H, last 8-12 meetings, venue bias, rivalry patterns. "
                "Cover professional criteria: team strength, recent form (5-10), head-to-head, xG/xGA, shot quality, PPDA, pressing, set-pieces, lineup quality, squad depth, rotation risk, fatigue, schedule density, rest days, travel, venue/home advantage, crowd intensity, tactical matchups, managerial styles, transitions, turnovers, goalkeeper quality, finishing variance, penalties, discipline risk, referee tendencies, weather, pitch, altitude, timezone, motivation/stakes, odds movement, market agreement, bookmaker margin, morale. "
                "Cite every source used (title + URL). Avoid fabrication."
            )
            prompt = (
                query_header + "\n" + requirements + "\n\n"
                "Return ONLY valid JSON with schema: {report:string, sources:[{title:string,url:string,pub_date?:string}], timestamp:string, window:{start:string,end:string}, history:{h2h_summary:string, last_meetings:[{date:string,competition:string,home:string,away:string,score:string}], venue_bias:string, long_term_trends:string}}. "
                "'report' should be a compact bullet summary of key findings."
            )
            # Not all SDKs expose a google_search tool; call Flash without tools if unavailable
            # Configure Google Search tool if SDK supports it
            tools = None
            if Tool and GoogleSearchRetrieval:
                try:
                    google_search_tool = Tool.from_google_search_retrieval(GoogleSearchRetrieval())
                    tools = [google_search_tool]
                except Exception:
                    tools = None
            resp = await self._generate_with(
                self.model_flash,
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
                tools=tools,
                tool_config=None,
            )
            data = self._extract_response_text(resp)
            if not data:
                return None
            import json
            try:
                obj = json.loads(data)
            except Exception:
                try:
                    clean = self._sanitize_json_string(data)
                    obj = json.loads(clean)
                except Exception:
                    return None
            # Filter sources by recency window if pub_date is present
            try:
                win = {"start": start_date, "end": end_date}
                sources = obj.get("sources") or []
                filtered = []
                for s in sources:
                    try:
                        pd = s.get("pub_date")
                        if not pd:
                            filtered.append(s)  # keep if no pub_date
                            continue
                        # Simple ISO date parse (YYYY-MM-DD)
                        y,m,d = int(pd[:4]), int(pd[5:7]), int(pd[8:10])
                        if f"{y:04d}-{m:02d}-{d:02d}" >= win["start"] and f"{y:04d}-{m:02d}-{d:02d}" <= win["end"]:
                            filtered.append(s)
                    except Exception:
                        filtered.append(s)
                obj["sources"] = filtered
                obj["window"] = win
            except Exception:
                pass
            return obj
        except Exception:
            logger.exception("Error during online grounded context gathering")
            return None

    async def predict_advanced(self, home_team: str, away_team: str, date: Optional[str] = None) -> Dict:
        """
        Rich LLM analysis with probabilities, prediction label, confidence, brief summary, and top factors.
        Output contract:
        {
          home_win:number, draw:number, away_win:number,
          prediction:string, confidence:number,
          summary:string,
          factors:[{name:string, impact:number, evidence:string}]
        }
        """
        if not self._client_ready:
            return {"error": "LLM not configured"}

        debug_mode = os.getenv("DEBUG_LLM") == "1"

        # 1) Gather up-to-date context via Gemini 2.5 Flash with online grounding
        context_blob = None
        try:
            context_blob = await self._gather_online_context(home_team, away_team, date)
        except Exception:
            logger.exception("Flash grounding failed; continuing without external context")

        # 2) Build the Pro prompt (with context if available)
        base_header = f"Matchup: {home_team} vs {away_team}.\n"
        context_header = ""
        if context_blob and isinstance(context_blob, dict):
            report = context_blob.get("report") or ""
            sources = context_blob.get("sources") or []
            timestamp = context_blob.get("timestamp") or ""
            # Keep prompt tight but grounded
            sources_lines = []
            for s in sources[:6]:
                try:
                    title = str(s.get("title", ""))[:120]
                    url = str(s.get("url", ""))[:200]
                    if title or url:
                        sources_lines.append(f"- {title} {url}")
                except Exception:
                    continue
            context_header = (
                "CONTEXT (latest online research):\n" + (report[:4000]) + "\n\n" +
                ("SOURCES:\n" + "\n".join(sources_lines) + "\n" if sources_lines else "") +
                (f"TIMESTAMP: {timestamp}\n" if timestamp else "") +
                "\nINSTRUCTIONS: Base your analysis strictly on the CONTEXT above; if uncertain or data is missing, say 'uncertain'. Do NOT fabricate.\n\n"
            )
        
        prompt = (
            base_header + context_header +
            "Return ONLY valid JSON per the instructed schema. "
            "Be brief but cutting-edge: max 5-8 lines in 'summary', and 8-12 'factors'."
        )

        try:
            resp = await self._generate_with(
                self.model_advanced,
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
            )
            data = self._extract_response_text(resp)
            if not data:
                return {"error": "Gemini returned no extractable data"}

            if debug_mode:
                try:
                    snippet = str(data).replace('\n',' ')[:200]
                    logger.warning(f"Gemini advanced raw payload (snippet): {snippet}")
                except Exception:
                    pass

            import json
            obj = None
            try:
                obj = json.loads(data)
            except Exception:
                data_clean = self._sanitize_json_string(data)
                try:
                    obj = json.loads(data_clean)
                except Exception:
                    return {"error": "LLM JSON parse failed (advanced)", "raw": str(data)[:200]}

            # Extract with coercion and normalization
            home = self._coerce_prob(obj.get("home_win", 33.3), 33.3)
            draw = self._coerce_prob(obj.get("draw", 33.3), 33.3)
            away = self._coerce_prob(obj.get("away_win", 33.4), 33.4)
            probs = self._normalize_probs(home, draw, away)

            prediction = obj.get("prediction")
            if not isinstance(prediction, str) or not prediction:
                if probs["home_win"] >= probs["draw"] and probs["home_win"] >= probs["away_win"]:
                    prediction = f"{home_team} Win"
                elif probs["away_win"] >= probs["draw"] and probs["away_win"] >= probs["home_win"]:
                    prediction = f"{away_team} Win"
                else:
                    prediction = "Draw"

            confidence = self._coerce_prob(obj.get("confidence", max(probs.values())), max(probs.values()))
            summary = obj.get("summary") or ""
            if not isinstance(summary, str):
                summary = str(summary)

            factors = obj.get("factors") or []
            cleaned_factors = []
            if isinstance(factors, list):
                for f in factors[:12]:
                    try:
                        name = str(f.get("name", "Factor"))
                        impact = float(self._coerce_prob(f.get("impact", 0), 0))
                        evidence = str(f.get("evidence", ""))
                        cleaned_factors.append({"name": name, "impact": impact, "evidence": evidence})
                    except Exception:
                        continue

            # Prepare top-5 sources from grounded context for transparency
            sources_for_result: List[Dict] = []
            try:
                if context_blob and isinstance(context_blob, dict):
                    _srcs = context_blob.get("sources") or []
                    for s in _srcs[:5]:
                        try:
                            title = str(s.get("title", ""))[:160]
                            url = str(s.get("url", ""))[:300]
                            if title or url:
                                sources_for_result.append({"title": title, "url": url})
                        except Exception:
                            continue
            except Exception:
                pass

            result = {
                **probs,
                "prediction": prediction,
                "confidence": confidence,
                "summary": summary,
                "factors": cleaned_factors,
                "sources": sources_for_result,
                "framework": "gemini-llm",
                "source": self.model_id,
                "raw": data,
            }
            return result
        except Exception:
            logger.exception("Advanced LLM prediction failed")
            return {"error": "Error generating advanced analysis"}
