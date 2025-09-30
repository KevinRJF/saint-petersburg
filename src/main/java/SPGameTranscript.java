import java.util.ArrayList;

public class SPGameTranscript {
	ArrayList<SPState> states;
	ArrayList<SPAction> actions;
	ArrayList<String> playerNames;

	public SPGameTranscript() {
		states = new ArrayList<>();
		actions = new ArrayList<>();
		playerNames = new ArrayList<>();
	}

	public void addState(SPState state) {
		states.add(state);
	}

	public void addAction(SPAction action) {
		actions.add(action);
	}

	public void addPlayerName(String playerName) {
		playerNames.add(playerName);
	}

	public ArrayList<SPState> getStates() {
		return states;
	}

	public ArrayList<SPAction> getActions() {
		return actions;
	}

	public ArrayList<String> getPlayerNames() {
		return playerNames;
	}

	public boolean[] getWinners() {
		return states.get(states.size() - 1).isWinner;
	}
}
