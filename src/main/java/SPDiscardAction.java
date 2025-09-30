public class SPDiscardAction extends SPAction {

	public SPDiscardAction(SPState state) {
		super(state);
	}

	@Override
	public SPState take(SPState state) {
		state.discardPile.add(state.observedCard);
		state.observedCard = null;
		// Advance the turn to the next player
		state.playerTurn = (state.playerTurn + 1) % state.numPlayers;
		return state;
	}

	@Override
	public String toString() {
		return String.format("Player %d discards the observed %s card.", player + 1, state.observedCard != null ? state.observedCard.name : "null");
	}
}
