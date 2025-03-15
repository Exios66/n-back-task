# The N-Back Task

## v0.0.1 Updates

### Key Enhancements and Debug Fixes

1. Unified Response Handling:
* The new global function processResponse(e) is used by both the keyboard and touch events. It checks if a trial is in progress, calculates the reaction time, determines if the response is correct, and then clears a scheduled timeout. This avoids duplicate code and makes the logic more robust.
2. Global Current Trial Storage:
* A global variable currentTrial now holds the trial information so that it’s available to the response handler regardless of input type.
3. Timeout Management:
* The response timeout (stored in responseTimeout) is cleared as soon as a valid response is received, preventing the omission handler from firing if the user responded in time.
4. Consistent UI Feedback:
* The display’s classes are updated to show “active,” “correct,” or “error” states, ensuring a smooth visual transition.
