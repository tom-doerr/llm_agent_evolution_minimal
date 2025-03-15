# Hypotheses for Fixing Agent Memory Assertions

1. **XML Tags in Memory String**
   - **Issue**: Memory storage includes raw XML tags from agent responses
   - **Evidence**: 
     - main.py line 73: `assert '<remember>' not in memory`
     - Agent.memory property joins MemoryItem outputs without XML sanitization
   - **Fix**: Use XML text extraction and HTML entity encoding
   - **Verification**: Added HTML entity encoding for XML tags in memory formatting
   - **Verification**:
     - Update MemoryItem output processing to use ET.tostring(method='text')
     - Add XML namespace removal to extract_xml()
   - **Verification**: Memory now shows plain text values without markup
   - **Code Impact**: Modify Agent.memory property formatting

2. **Shell Command Test Responses**
   - **Issue**: Test mode responses missing required XML structure
   - **Evidence**:
     - main.py line 136 expects 'plexsearch.log' in cleaned output
     - _handle_shell_commands returns raw string instead of XML-wrapped
   - **Fix**: Maintain exact XML whitespace formatting from main.py tests
   - **Verification**: Formatted test response XML with proper indentation
   - **Verification**:
     - Keep <shell> tags in test responses
     - Add proper <message> wrapping for assertions
   - **Verification**: Test responses now match assertion requirements
   - **Code Impact**: Update _handle_shell_commands test responses

3. **Memory Combination Logic**
   - **Issue**: mate() method duplicates parent memories
   - **Evidence**:
     - main.py line 208-209 requires both 321 and 477 in combined memory
     - Current implementation uses string-based deduplication
   - **Fix**: Use MemoryItem equality checks with normalized values
   - **Verification**:
     - Update mate() to use MemoryItem instances in seen set
     - Verify MemoryItem.__eq__ compares normalized fields
   - **Validation**: Current implementation compares normalized fields

4. **Context Instructions Filtering**
   - **Issue**: Core instructions might leak to memory
   - **Evidence**: main.py asserts context not in memory
   - **Fix**: Strict type filtering in memory property
   - **Verification**: _context_instructions are stored separately
