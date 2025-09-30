import java.util.ArrayList;
/* Models a choice to use an observatory to draw a random card from a given deck with more than one card. */
public class SPObserveDeckAction extends SPAction {

	public final int deckIndex; // Index of the deck to observe

	SPObserveDeckAction(SPState state, int deckIndex) {
		super(state);
		this.deckIndex = deckIndex;
	}
	
	@Override
	public SPState take(SPState state) {
		// Player did not pass
		state.playerPassed[player] = false;
		ArrayList<SPCard> deck = null;
		switch (deckIndex) {
			case 0: // Worker Deck
				deck = state.workerDeck;
				break;
			case 1: // Building Deck
				deck = state.buildingDeck;
				break;
			case 2: // Aristocrat Deck
				deck = state.aristocratDeck;
				break;
			case 3: // Trading Deck
				deck = state.tradingDeck;
				break;
		}
		if (deck == null) {
			throw new IllegalArgumentException("Invalid deck index: " + deckIndex);
		}
		// Observe the top card of the selected deck
		state.observedCard = drawRandomCard(deck);
		state.usedObservatories[state.playerTurn]++; // Increment the number of observatories used by the player
		return state;
	}

	@Override
	public String toString() {
		String[] deckNames = {"worker deck", "building deck", "aristocrat deck", "trading deck"};
		if (deckIndex < 0 || deckIndex >= deckNames.length) {
			throw new IllegalArgumentException("Invalid deck index: " + deckIndex);
		}
		return String.format("Player %d observes %s.", player + 1, deckNames[deckIndex]);
	}
}
