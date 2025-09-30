public abstract class SPPlayer {
	private String name;

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
}
