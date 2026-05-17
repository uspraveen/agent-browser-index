# Updated methods for browser_controller.py
# These need to replace the existing methods

def _get_large_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
    """Get large note by ID using index for fast O(1) lookup."""
    # Try index-based retrieval first (fast path)
    if note_id in self.large_notes_index:
        try:
            metadata = self.large_notes_index[note_id]
            line_number = metadata["file_line_number"]
            
            # Read specific line from JSONL (fast seeking)
            with open(self.large_notes_path, "r", encoding="utf-8") as f:
                for current_line_num, raw_line in enumerate(f, start=1):
                    if current_line_num == line_number:
                        try:
                            return json.loads(raw_line.strip())
                        except json.JSONDecodeError:
                            print(f"⚠️  Corrupted JSONL line {line_number} for note {note_id}")
                            break
        except Exception as e:
            print(f"⚠️  Index-based retrieval failed for {note_id}: {e}. Falling back to scan.")
    
    # Fallback: Scan JSONL if index lookup failed
    entries = self._load_large_note_entries()
    for entry in reversed(entries):
        if str(entry.get("id", "")).strip() == str(note_id).strip():
            return entry
    return None

async def _list_large_notes(self, limit: Any = 20, newest_first: Any = True) -> ActionResult:
    """List metadata for large notes using index (fast)."""
    try:
        safe_limit = max(1, min(200, int(limit)))
        if isinstance(newest_first, bool):
            is_newest_first = newest_first
        else:
            is_newest_first = str(newest_first).strip().lower() in {"1", "true", "yes", "y"}

        # Use index instead of scanning JSONL
        notes_list = list(self.large_notes_index.values())
        if is_newest_first:
            notes_list = list(reversed(notes_list))
        
        selected = notes_list[:safe_limit]

        output = json.dumps(
            {
                "path": str(self.large_notes_path),
                "index_path": str(self.large_notes_index_path),
                "total_notes": len(self.large_notes_index),
                "returned": len(selected),
                "newest_first": is_newest_first,
                "notes": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
        return ActionResult(
            success=True,
            action_type=ActionType.LIST_LARGE_NOTES,
            description=f"Listed {len(selected)} large notes",
            output=output,
        )
    except Exception as e:
        return ActionResult(
            success=False,
            action_type=ActionType.LIST_LARGE_NOTES,
            description="List large notes",
            error=f"Failed to list large notes from index: {str(e)}",
        )

async def _search_large_notes(self, query: str, limit: Any = 10) -> ActionResult:
    """Search large notes using index for metadata, with fallback to content search."""
    try:
        q = str(query or "").strip()
        if not q:
            return ActionResult(
                success=False,
                action_type=ActionType.SEARCH_LARGE_NOTES,
                description="Search large notes",
                error="query must not be empty.",
            )

        safe_limit = max(1, min(100, int(limit)))
        q_lower = q.lower()

        # Phase 1: Search index metadata (fast)
        matches = []
        for note_id, metadata in self.large_notes_index.items():
            match_score = 0
            if q_lower in metadata.get("title", "").lower():
                match_score += 10
            if q_lower in metadata.get("contains", "").lower():
                match_score += 8
            if q_lower in metadata.get("summary", "").lower():
                match_score += 5
            if q_lower in metadata.get("source_domain", "").lower():
                match_score += 3
            if q_lower in metadata.get("why", "").lower():
                match_score += 2
            
            if match_score > 0:
                matches.append((match_score, metadata))

        # Sort by relevance
        matches.sort(reverse=True, key=lambda x: x[0])
        results = [m[1] for m in matches[:safe_limit]]

        # Phase 2: If metadata search yields few results, search content
        if len(results) < safe_limit // 2:
            print(f"   Metadata search found {len(results)} matches, searching content...")
            entries = self._load_large_note_entries()
            for entry in entries:
                if len(results) >= safe_limit:
                    break
                note_id = entry.get("id")
                # Skip if already matched
                if any(r["id"] == note_id for r in results):
                    continue
                    
                if q_lower in str(entry.get("content", "")).lower():
                    if note_id in self.large_notes_index:
                        results.append(self.large_notes_index[note_id])

        output = json.dumps(
            {
                "path": str(self.large_notes_path),
                "query": q,
                "total_matches": len(results),
                "returned": len(results),
                "notes": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        return ActionResult(
            success=True,
            action_type=ActionType.SEARCH_LARGE_NOTES,
            description=f"Found {len(results)} matching notes",
            output=output,
        )
    except Exception as e:
        return ActionResult(
            success=False,
            action_type=ActionType.SEARCH_LARGE_NOTES,
            description="Search large notes",
            error=f"Search failed: {str(e)}",
        )
