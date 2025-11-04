import java.util.ArrayList;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public abstract class SPAction {

	protected SPState state; // a state for which this is a legal action
	protected int player; // The player who is taking the action
	static RandomGenerator cardRandom = RandomGenerator.of("Xoroshiro128PlusPlus");
	// Java's default Random uses a linear congruential generator which would have cross-correlations
	// for sequential seeds used in MCTS for chance nodes.

	// Construct an action based on a state for which it is legal
	SPAction(SPState state) {
		this.state = state;
		this.player = state.playerTurn;
	}

	// Take the action on the given state, returning the resulting state.
	abstract public SPState take(SPState state); // assumes no deep cloning of state, returns resulting state
	
	// Check if this action leads to a chance event
	public boolean isChanceAction() {
		return this instanceof SPPossibleChanceAction && ((SPPossibleChanceAction) this).isChanceAction();
	}

	// Take the action on the state for which it is legal, returning the resulting state.
	public SPState take() {
		return take(state);
	}

	// Take the action given a random generator seed
	public SPState take(long seed) {
		RandomGenerator preservedCardRandom = cardRandom;
		cardRandom = RandomGeneratorFactory.of("Xoroshiro128PlusPlus").create(seed);
		SPState result = take();
		cardRandom = preservedCardRandom;
		return result;
	}

	// Take the action on the state for which it is legal, 
	// either on a clone of the state or on the state itself depending on the onClone parameter.
	public SPState take(boolean onClone) { // onClone indicates whether or not action will be taken on
		return take(onClone ? (SPState) state.clone() : state);
	}

	public SPCard drawRandomCard(ArrayList<SPCard> deck) {
		if (deck.isEmpty()) {
			return null; // No cards to draw
		}
		int index = cardRandom.nextInt(deck.size());
		SPCard card = deck.get(index);
		if (index == deck.size() - 1) {
			deck.remove(index);
		}
		else {
			deck.set(index, deck.get(deck.size() - 1));
			deck.remove(deck.size() - 1);
		}
		return card;
	}

	public void refillTopRow(ArrayList<SPCard> deck) {
		int numCardsToDraw = SPState.MARKET_SIZE - state.upperCardRow.size() - state.lowerCardRow.size();
		for (int i = 0; i < numCardsToDraw && !deck.isEmpty(); i++) {
			state.upperCardRow.add(drawRandomCard(deck));
		}
	}
}