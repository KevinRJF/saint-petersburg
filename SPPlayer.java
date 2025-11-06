public abstract class SPPlayer {
	protected String name;
	public static final long UNKNOWN_TIME = -1;
	protected long timeRemainingMillis = UNKNOWN_TIME;

	public SPPlayer(String name) {
		// Make into a legal java identifier by removing non-alphanumeric characters and replacing whitespace with underscores
		this.name = name.replaceAll("[^a-zA-Z0-9\\s]", "").replaceAll("\\s+", "_");
		// Ensure the name is not empty
		if (this.name.isEmpty()) {
			this.name = "Unnamed_Player";
		}
	}

	public String getName() {
		return name;
	}

	// Return the index of the chosen action from state.getLegalActions()
	// This method should be implemented by subclasses to define the player's strategy.
	public abstract int getAction(SPState state);

	public int getAction(SPState state, long timeRemainingMillis) {
		// Default implementation ignores time limit
		// For the last competition, players should override this method and respect the time limit.
		// If not overridden, the time limit will be ignored, and the player may be disqualified.
		this.timeRemainingMillis = timeRemainingMillis;
		return getAction(state);
	}
}
