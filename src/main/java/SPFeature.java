public abstract class SPFeature {

	private String name;
	private String description;

	public SPFeature(String name, String description) {
		// Ensure that the name is not null, has whitespace replaced by underscores, and has non-alphanumeric characters removed
		if (name == null || name.trim().isEmpty()) {
			throw new IllegalArgumentException("Feature name cannot be null or empty");
		}
		this.name = name.replaceAll("\\s+", "_").replaceAll("[^a-zA-Z0-9_]", "");
		this.description = description;
	}

	public abstract Object getValue(SPState state);

	public String getName() {
		return name;
	}

	public String getDescription() {
		return description;
	}

	@Override
	public String toString() {
		return name;
	}
}
