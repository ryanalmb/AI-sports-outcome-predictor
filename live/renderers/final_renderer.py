"""
Final renderer for Flash Live Degen feature.
This module formats the final degen-toned report for Telegram.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class FinalRenderer:
    """
    Renderer for the final degen-toned report.
    """
    
    def __init__(self):
        """
        Initialize the FinalRenderer.
        """
        pass
    
    def format_final_report(self, analysis_data: Dict[str, Any]) -> str:
        """
        Format the final degen-toned report.
        
        Args:
            analysis_data: Dictionary containing all analysis data
            
        Returns:
            Formatted string for Telegram
        """
        # Extract data
        event_query = analysis_data.get("event_query", "Unknown Event")
        sources = analysis_data.get("sources", [])
        processed_data = self._aggregate_processed_data(sources)
        
        # Build report sections
        playbook = self._format_degen_playbook(processed_data, event_query)
        yolo_factors = self._format_yolo_factors(processed_data)
        odds_vibes = self._format_odds_vibes(processed_data)
        rug_checks = self._format_rug_checks(processed_data)
        entry_exit = self._format_entry_exit_spin()
        hfsp_hook = self._format_hfsp_hook()
        source_list = self._format_sources(sources)
        
        # Combine all sections
        report = (
            f"🏈 *DEGEN PLAYBOOK: {event_query}* (NFA)\n\n"
            f"{playbook}\n\n"
            f"🔥 *YOLO FACTORS*\n"
            f"{yolo_factors}\n\n"
            f"📊 *ODDS & MARKET VIBES*\n"
            f"{odds_vibes}\n\n"
            f"⚠️ *RUG CHECKS*\n"
            f"{rug_checks}\n\n"
            f"⏰ *ENTRY/EXIT SPIN*\n"
            f"{entry_exit}\n\n"
            f"💸 *HFSP HOOK*\n"
            f"{hfsp_hook}\n\n"
            f"📚 *SOURCES*\n"
            f"{source_list}"
        )
        
        return report
    
    def _aggregate_processed_data(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate processed data from all sources.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Aggregated data dictionary
        """
        aggregated = {
            "summaries": [],
            "key_points": [],
            "injuries_suspensions": [],
            "odds_movement": [],
            "confidence_scores": []
        }
        
        for source in sources:
            processed_data = source.get("processed_data", {})
            if processed_data:
                # Collect summaries
                summary = processed_data.get("summary", "")
                if summary:
                    aggregated["summaries"].append(summary)
                
                # Collect key points
                key_points = processed_data.get("key_points", [])
                aggregated["key_points"].extend(key_points)
                
                # Collect injuries/suspensions
                injuries = processed_data.get("injuries_suspensions", [])
                aggregated["injuries_suspensions"].extend(injuries)
                
                # Collect odds movement
                odds = processed_data.get("odds_movement", "")
                if odds:
                    aggregated["odds_movement"].append(odds)
                
                # Collect confidence scores
                confidence = processed_data.get("confidence", 0)
                aggregated["confidence_scores"].append(confidence)
        
        return aggregated
    
    def _format_degen_playbook(self, processed_data: Dict[str, Any], event_query: str) -> str:
        """
        Format the "Degen Playbook" section.
        
        Args:
            processed_data: Aggregated processed data
            event_query: The event query
            
        Returns:
            Formatted playbook section
        """
        summaries = processed_data.get("summaries", [])
        if not summaries:
            return "No clear thesis yet - market's playing coy! 🤷‍♂️"
        
        # Take the first summary as the main thesis
        main_thesis = summaries[0] if summaries else "The vibes are murky - proceed with caution!"
        return f"_{main_thesis}_"
    
    def _format_yolo_factors(self, processed_data: Dict[str, Any]) -> str:
        """
        Format the "YOLO Factors" section.
        
        Args:
            processed_data: Aggregated processed data
            
        Returns:
            Formatted YOLO factors section
        """
        key_points = processed_data.get("key_points", [])
        injuries = processed_data.get("injuries_suspensions", [])
        
        # Combine key points and injuries
        all_factors = []
        all_factors.extend(key_points[:3])  # Top 3 key points
        all_factors.extend(injuries[:2])     # Top 2 injuries/suspensions
        
        if not all_factors:
            return "Market's tight - no clear catalysts spotted yet! 🕵️‍♂️"
        
        # Format as bullet points (max 5)
        factors = all_factors[:5]
        formatted_factors = "\n".join([f"• {factor}" for factor in factors])
        return formatted_factors
    
    def _format_odds_vibes(self, processed_data: Dict[str, Any]) -> str:
        """
        Format the "Odds & Market Vibes" section.
        
        Args:
            processed_data: Aggregated processed data
            
        Returns:
            Formatted odds and market vibes section
        """
        odds_data = processed_data.get("odds_movement", [])
        confidence_scores = processed_data.get("confidence_scores", [])
        
        if not odds_data:
            return "Lines are holding steady - no major movement detected! 📉"
        
        # Get average confidence
        avg_confidence = 0
        if confidence_scores:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Format odds information
        odds_info = "\n".join([f"• {odds}" for odds in odds_data[:3]])  # Top 3 odds info
        return f"_{odds_info}_\n\n*Confidence Score*: {avg_confidence:.1f}/100"
    
    def _format_rug_checks(self, processed_data: Dict[str, Any]) -> str:
        """
        Format the "Rug Checks" section.
        
        Args:
            processed_data: Aggregated processed data
            
        Returns:
            Formatted rug checks section
        """
        # For now, we'll use generic rug checks since we don't have specific data
        # In a real implementation, this would come from the analysis
        return (
            "• Weather conditions could impact play\n"
            "• Late injury reports might not be reflected\n"
            "• Market manipulation possible in low-volume periods\n"
            "• Travel fatigue for away teams not fully accounted"
        )
    
    def _format_entry_exit_spin(self) -> str:
        """
        Format the "Entry/Exit Spin" section.
        
        Returns:
            Formatted entry/exit spin section
        """
        return (
            "_Timing is everything in degeneracy!_\n\n"
            "• *Entry*: Place early for best lines\n"
            "• *Add*: If initial catalysts confirm\n"
            "• *Take*: On first sign of narrative shift\n"
            "• *Bail*: If core thesis crumbles\n\n"
            "_Remember: Markets are unpredictable - always NFA!_ 🎯"
        )
    
    def _format_hfsp_hook(self) -> str:
        """
        Format the "HFSP Hook" section.
        
        Returns:
            Formatted HFSP hook section
        """
        return (
            "_If you print, consider yeeting a slice back into the treasury_\n"
            "— keeps the degen machine humming (NFA) 💸\n\n"
            "• Support the bot development\n"
            "• Fuel more spicy predictions\n"
            "• Keep the memecoin flowing\n\n"
            "_Not financial advice, just good vibes!_ 🚀"
        )
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """
        Format the sources section.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Formatted sources section
        """
        if not sources:
            return "No sources found during research! 🤷‍♂️"
        
        # Format as a list of domains/URLs
        source_lines = []
        for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources
            url = source.get("url", "")
            title = source.get("title", "")
            
            # Use title if available, otherwise use URL
            display_text = title if title else url
            source_lines.append(f"{i}\\. [{display_text}]({url})")
        
        return "\n".join(source_lines)