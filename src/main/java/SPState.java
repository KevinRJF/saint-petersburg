/* SPState - Represents a state of the board game Saint Petersburg */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class SPState implements Cloneable {

	public static final int EDITION = SPCard.EDITION; // Edition number for the game
	public static final int WORKER = 0; // Phase for worker cards
	public static final int BUILDING = 1; // Phase for building cards
	public static final int ARISTOCRAT = 2; // Phase for aristocrat cards
	public static final int TRADING = 3; // Phase for trading cards
	public static final int NUM_DECKS = 4; // Total number of phases in the game
	public static final int PUB_ACTION = 4; // Special psuedo-phase for the pub decisions
	public static final int END = 5; // End phase of the game
	public static final int INITIAL_RUBLES = 25; // Initial rubles for each player
	public static final List<String> PHASE_NAMES = Collections.unmodifiableList(Arrays.asList("Worker", "Building", "Aristocrat", "Trading", "Pub", "End")); // Names of the phases
	public static final List<Integer> UNIQUE_ARISTOCRAT_BONUS_POINTS = Collections.unmodifiableList(Arrays.asList(0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55)); // Bonus points for number of unique aristocrats
	// Bonus points for number of unique aristocrats at game end
	public static final int MAX_UNIQUE_ARISTOCRATS = UNIQUE_ARISTOCRAT_BONUS_POINTS.size() - 1; // Maximum number of unique aristocrats
	public int numPlayers = 2; // Number of players in the game
	public int round = 1; // Current round of the game
	public int phase = WORKER;
	public int playerTurn = 0; // Current player's turn
	public ArrayList<SPCard> workerDeck = new ArrayList<>(); // Deck of worker cards
	public ArrayList<SPCard> buildingDeck = new ArrayList<>(); // Deck of building cards
	public ArrayList<SPCard> aristocratDeck = new ArrayList<>(); // Deck of aristocrat cards
	public ArrayList<SPCard> tradingDeck = new ArrayList<>(); // Deck of trading cards
	public ArrayList<ArrayList<SPCard>> playerHands = new ArrayList<>(); // Hands of players
	public ArrayList<ArrayList<SPCard>> playerWorkers = new ArrayList<>(); // Workers of players 
	public ArrayList<ArrayList<SPCard>> playerBuildings = new ArrayList<>(); // Buildings of players
	public ArrayList<ArrayList<SPCard>> playerAristocrats = new ArrayList<>(); // Aristocrats of players
	public ArrayList<SPCard> upperCardRow = new ArrayList<>(); // Upper card row
	public ArrayList<SPCard> lowerCardRow = new ArrayList<>(); // Lower card row
	public ArrayList<SPCard> discardPile = new ArrayList<>(); // Discard pile for cards
	public int[] playerRubles = new int[numPlayers]; // Rubles of players
	public int[] playerPoints = new int[numPlayers]; // Points of players
	public boolean[] playerPassed = new boolean[numPlayers]; // Whether players have just passed
	public int[] startingPlayer = new int[NUM_DECKS]; // Starting player for each player 
	public int[] usedObservatories = new int[numPlayers]; // Used observatories
	public SPCard observedCard = null; // Card observed by the player
	public boolean[] isWinner = new boolean[numPlayers]; // Whether players are winners

	public SPState(int numPlayers) {
		this.numPlayers = numPlayers;
		if (numPlayers < 2 || numPlayers > 4) {
			throw new IllegalArgumentException("Number of players must be between 2 and 4.");
		}
		initialize();
	}

	public SPState() {
		initialize();
	}

	private void initialize() {
		// Initialize decks with cards from SPCard class
		for (SPCard card : SPCard.ALL_CARDS) {
			if (card.isTrading) {
				for (int i = 0; i < card.quantity; i++) {
					tradingDeck.add(card);
				}
			} else if (card.isWorker) {
				for (int i = 0; i < card.quantity; i++) {
					workerDeck.add(card);
				}
			} else if (card.isBuilding) {
				for (int i = 0; i < card.quantity; i++) {
					buildingDeck.add(card);
				}
			} else if (card.isAristocrat) {
				for (int i = 0; i < card.quantity; i++) {
					aristocratDeck.add(card);
				}
			}
		}
		// Shuffle decks
		Collections.shuffle(workerDeck);
		Collections.shuffle(buildingDeck);
		Collections.shuffle(aristocratDeck);	
		Collections.shuffle(tradingDeck);
		// Initial rubles for each player
		Arrays.fill(playerRubles, INITIAL_RUBLES);
		// Initialize upper card row
		int initialWorkers = 2 * numPlayers;
		for (int i = 0; i < initialWorkers; i++) {
			upperCardRow.add(workerDeck.remove(workerDeck.size() - 1));
		}
		// Initial starting roles for each player
		ArrayList<Integer> startingPlayersList = new ArrayList<>();
		for (int i = 0; i < NUM_DECKS; i++) {
			startingPlayersList.add(i % numPlayers);
		}
		Collections.shuffle(startingPlayersList);
		for (int i = 0; i < NUM_DECKS; i++) {
			startingPlayer[i] = startingPlayersList.get(i);
		}
		playerTurn = startingPlayer[WORKER]; // Set the player to start the worker phase of the first round
		// Initialize player hands, workers, buildings, and aristocrats
		for (int i = 0; i < numPlayers; i++) {
			playerHands.add(new ArrayList<>());
			playerWorkers.add(new ArrayList<>());
			playerBuildings.add(new ArrayList<>());
			playerAristocrats.add(new ArrayList<>());
		}
	}

	public boolean isGameOver() {
		return phase == END;
	}

	private <T> String listToStringNoBrackets(List<T> list) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < list.size(); i++) {
			sb.append(list.get(i).toString());
			if (i < list.size() - 1) {
				sb.append(", ");
			}
		}
		return sb.toString();
	}

	public int getNumUniqueAristocrats(int player) {
		// Count unique aristocrats for a player
		return new HashSet<>(playerAristocrats.get(player)).size();
	}

	public ArrayList<Integer> getStartingPhases(int player) {
		// Count starting phases for a player
		ArrayList<Integer> startingPhases = new ArrayList<>();
		for (int phase = 0; phase < NUM_DECKS; phase++) {
			if (startingPlayer[phase] == player) {
				startingPhases.add(phase);
			}
		}
		return startingPhases;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		if (isGameOver()) {
			sb.append(String.format("Round: %d, Phase: %s, Game Over\n", round, PHASE_NAMES.get(phase)));
		} else {
			sb.append(String.format("Round: %d, Phase: %s, Player Turn: %d\n", round, PHASE_NAMES.get(phase), playerTurn + 1));
		}
		sb.append("Player\tPoints (per phase)\tRubles (per phase)\tAristocrats\tStarting Phases\n");
		for (int player = 0; player < numPlayers; player++) {
			int[] pointsPerPhase = new int[3];
			for (int phase = 0; phase < pointsPerPhase.length; phase++) {
				switch (phase) {
					case WORKER:
						pointsPerPhase[phase] = playerWorkers.get(player).stream().mapToInt(card -> card.points).sum();
						break;
					case BUILDING:
						pointsPerPhase[phase] = playerBuildings.get(player).stream().mapToInt(card -> card.points).sum();
						break;
					case ARISTOCRAT:
						pointsPerPhase[phase] = playerAristocrats.get(player).stream().mapToInt(card -> card.points).sum();
						break;
				}
			}
			int[] rublesPerPhase = new int[NUM_DECKS];
			for (int phase = 0; phase < NUM_DECKS; phase++) {
				switch (phase) {
					case WORKER:
						rublesPerPhase[phase] = playerWorkers.get(player).stream().mapToInt(card -> card.rubles).sum();
						break;
					case BUILDING:
						rublesPerPhase[phase] = playerBuildings.get(player).stream().mapToInt(card -> card.rubles).sum();
						break;
					case ARISTOCRAT:
						rublesPerPhase[phase] = playerAristocrats.get(player).stream().mapToInt(card -> card.rubles).sum();
						break;
				}
			}
			// Convert the getStartingPhases integers into a list of Strings representation
			List<String> startingPhaseStrings = new ArrayList<>();
			for (int startingPhase : getStartingPhases(player)) {
				startingPhaseStrings.add(PHASE_NAMES.get(startingPhase));
			}
			sb.append(String.format("%6d\t%6d (%d, %d, %d)\t%6d (%d, %d, %d)\t%11d\t%s\n",
					player + 1, playerPoints[player], pointsPerPhase[WORKER], pointsPerPhase[BUILDING], pointsPerPhase[ARISTOCRAT],
					playerRubles[player], rublesPerPhase[WORKER], rublesPerPhase[BUILDING], rublesPerPhase[ARISTOCRAT], 
					getNumUniqueAristocrats(player), listToStringNoBrackets(startingPhaseStrings)));
		}
		sb.append(String.format("Deck Sizes: Worker: %d, Building: %d, Aristocrat: %d, Trading: %d\n",
				workerDeck.size(), buildingDeck.size(), aristocratDeck.size(), tradingDeck.size()));
		sb.append("\nUpper Card Row: ").append(listToStringNoBrackets(upperCardRow)).append("\n");
		sb.append("Lower Card Row: ").append(listToStringNoBrackets(lowerCardRow)).append("\n");

		for (int p = 0; p < numPlayers; p++) {
			sb.append("\nPlayer " + (p + 1) + ":\n");
			sb.append(String.format("Hand: %s\n", listToStringNoBrackets(playerHands.get(p))));	
			sb.append(String.format("Workers: %s\n", listToStringNoBrackets(playerWorkers.get(p))));
			sb.append(String.format("Buildings: %s\n", listToStringNoBrackets(playerBuildings.get(p))));
			sb.append(String.format("Aristocrats: %s\n", listToStringNoBrackets(playerAristocrats.get(p))));
		}	

		return sb.toString();
	}

	@SuppressWarnings("unchecked")
	public SPState clone() {
		try {
			SPState copy = (SPState) super.clone(); // Shallow copy
			copy.workerDeck = (ArrayList<SPCard>) workerDeck.clone();
			copy.buildingDeck = (ArrayList<SPCard>) buildingDeck.clone();
			copy.aristocratDeck = (ArrayList<SPCard>) aristocratDeck.clone();
			copy.tradingDeck = (ArrayList<SPCard>) tradingDeck.clone();
			copy.playerHands = new ArrayList<ArrayList<SPCard>>();
			copy.playerWorkers = new ArrayList<ArrayList<SPCard>>();
			copy.playerBuildings = new ArrayList<ArrayList<SPCard>>();
			copy.playerAristocrats = new ArrayList<ArrayList<SPCard>>();
			for (int i = 0; i < playerHands.size(); i++) {
				copy.playerHands.add((ArrayList<SPCard>) playerHands.get(i).clone());
				copy.playerWorkers.add((ArrayList<SPCard>) playerWorkers.get(i).clone());
				copy.playerBuildings.add((ArrayList<SPCard>) playerBuildings.get(i).clone());
				copy.playerAristocrats.add((ArrayList<SPCard>) playerAristocrats.get(i).clone());
			}
			copy.upperCardRow = (ArrayList<SPCard>) upperCardRow.clone();
			copy.lowerCardRow = (ArrayList<SPCard>) lowerCardRow.clone();
			copy.discardPile = (ArrayList<SPCard>) discardPile.clone();
			copy.playerRubles = playerRubles.clone();
			copy.playerPoints = playerPoints.clone();
			copy.playerPassed = playerPassed.clone();
			copy.startingPlayer = startingPlayer.clone();
			copy.usedObservatories = usedObservatories.clone();
			copy.isWinner = isWinner.clone();
			return copy;
		} catch (CloneNotSupportedException e) {
			throw new AssertionError(); // Can't happen
		}
	}

	public ArrayList<SPAction> getLegalActions() {
		ArrayList<SPAction> legalActions = new ArrayList<>();
		if (isGameOver()) {
			return legalActions; // No actions allowed in the end phase
		}

		// Determine if the player has room in their hand for another card
		// Check if the player has a Warehouse building card
		boolean hasWarehouse = playerBuildings.get(playerTurn).stream().anyMatch(b -> b.name.equals("Warehouse"));
		boolean hasRoomInHand = (hasWarehouse ? 4 : 3) - playerHands.get(playerTurn).size() > 0; // Check if the player has room in hand

		// If the player has bought the "Carpenter Workshop", the building costs are reduced by 1
		boolean hasBuildingDiscount = false;
		if (playerBuildings.get(playerTurn).stream()
				.anyMatch(c -> c.name.equals("Carpenter Workshop"))) {
			hasBuildingDiscount = true;
		}

		// If the player has bought the "Gold Smelter", the aristocrat costs are reduced by 1
		boolean hasAristocratDiscount = false;
		if (playerBuildings.get(playerTurn).stream()
				.anyMatch(c -> c.name.equals("Gold Smelter"))) {
			hasAristocratDiscount = true;
		}

		// Determine if the player has an unused observatory
		int numUnusedObservatories = (int) playerBuildings.get(playerTurn).stream()
				.filter(observatory -> observatory.name.equals("Observatory")).count();
		numUnusedObservatories -= usedObservatories[playerTurn]; // Subtract the number of observatories used this round

		// If they have observed a pile, they must choose what to do with the observed card.
		if (observedCard != null) {
			// Discard the observed card
			legalActions.add(new SPDiscardAction(this));
			// If the player has room in hand, they can add the observed card to their hand
			if (hasRoomInHand) {
				legalActions.add(new SPAddToHandAction(this, observedCard, null)); // null indicates the card is not from a row
			}
			// If the player can afford to buy the observed card, add buy action(s)
			int rubles = playerRubles[playerTurn];
			if (rubles > 1) { // all buys require at least 1 ruble) {
				SPCard card = observedCard; // The card being observed
				int cost = card.cost;
				if (card.isBuilding && hasBuildingDiscount) {
					cost--; // Building costs are reduced by 1 if the player has the Carpenter Workshop
				}
				if (card.isAristocrat && hasAristocratDiscount) {
					cost--; // Aristocrat costs are reduced by 1 if the player has the Gold Smelter
				}

				// Determine the destination for the card
				ArrayList<SPCard> cardDestination = null;
				if (card.isWorker) {
					cardDestination = playerWorkers.get(playerTurn);
				} else if (card.isBuilding) {
					cardDestination = playerBuildings.get(playerTurn);
				} else if (card.isAristocrat) {
					cardDestination = playerAristocrats.get(playerTurn);
				}
				// Reduce the cost by 1 for each same card in the destination
				for (SPCard c : cardDestination) {
					if (c == card) {
						cost--;
					}
				}

				// If the card is not a trading card, create a buy action if affordable
				if (!card.isTrading) {
					if (cost < 1) {
						cost = 1; // Ensure the cost is not less than 1
					}
					if (rubles >= cost) {
						// Create a buy action for the card
						SPBuyAction buyAction = new SPBuyAction(this, card, null, cardDestination, cost);
						legalActions.add(buyAction);
					}
				} // end observed non-trading card handling 
				else { // If the card is a trading card, create a buy action(s) with replaced card(s)
					// Compute the replaceable cards with associated costs
					Map<SPCard, Integer> replaceableCards = new HashMap<>();
					for (SPCard c : cardDestination) {
						if (c.isTrading) {
							continue; // cannot replace trading cards with trading cards
						}
						// Worker trading cards have to replace compatible worker cards
						if (c.isWorker) {
							// determine if the trading card can replace the worker card
							if (!c.name.equals("Czar and Carpenter")) // automatically compatible
							{
								String cardNotes = card.other;
								// if the card notes has the string "replaces ",
								// then extract the trading card name thereafter to the end of the string
								int index = cardNotes.indexOf("replaces ");
								String replacedWorkerType = cardNotes.substring(index + 9).trim();
								if (!c.name.equalsIgnoreCase(replacedWorkerType)) {
									continue; // incompatible worker card
								}
							}
						}
						int replaceCost = cost - c.cost; // Cost of replacing the card
						if (c.name.equals("Observatory")) {
							if (numUnusedObservatories < 1) {
								continue; // Cannot replace an Observatory if none are left unused
							}
						}
						if (card.name.endsWith("Village")) {
							replaceCost -= 4; // Potjomkin's/Potemkin Village credits 6 while only costing 2
						}
						if (replaceCost < 1) {
							replaceCost = 1; // Ensure the cost is not less than 1
						}
						if (rubles >= replaceCost) {
							replaceableCards.put(c, replaceCost); // Add the card and its cost to the map
						}
					}
					// Create a buy action for each replaceable card
					for (Map.Entry<SPCard, Integer> entry : replaceableCards.entrySet()) {
						SPCard replacedCard = entry.getKey();
						int replaceCost = entry.getValue();
						SPBuyAction buyAction = new SPBuyAction(this, card, null, cardDestination, replaceCost, replacedCard);
						legalActions.add(buyAction);
					}
				} // end observed trading card handling
			} // end has at least 1 ruble to buy observed card
		} // end observed card handling
		else if (phase == PUB_ACTION) {
			// Determine the number of pubs the player has
			int numPubs = (int) playerBuildings.get(playerTurn).stream().filter(c -> c.name.equals("Pub")).count();
			int maxPoints = numPubs * 5; // Each Pub can purchase up to 5 points at 2 rubles/point
			int rubles = playerRubles[playerTurn];
			for (int points = 0; points <= maxPoints && 2 * points <= rubles; points++) {
				legalActions.add(new SPPubPointsAction(this, points));
			}
		}
		// Otherwise, is it a normal pass/observe/buy/add-to-hand decision.
		else {
			
			// Add pass action
			legalActions.add(new SPPassAction(this));
			
			// Add buy actions
			int rubles = playerRubles[playerTurn];

			if (rubles > 0) { // all buys require at least 1 ruble

				ArrayList<ArrayList<SPCard>> cardSources = new ArrayList<>();
				cardSources.add(playerHands.get(playerTurn)); // Player's hand
				cardSources.add(upperCardRow);
				cardSources.add(lowerCardRow);

				for (ArrayList<SPCard> cardSource : cardSources) {
					for (SPCard card : cardSource) {
						int cost = card.cost;
						if (cardSource == lowerCardRow) {
							cost--; // Lower row cards cost 1 less
						}
						if (card.isBuilding && hasBuildingDiscount) {
							cost--; // Building costs are reduced by 1 if the player has the Carpenter Workshop
						}
						if (card.isAristocrat && hasAristocratDiscount) {
							cost--; // Aristocrat costs are reduced by 1 if the player has the Gold Smelter
						}
						// Determine the destination for the card
						ArrayList<SPCard> cardDestination = null;
						if (card.isWorker) {
							cardDestination = playerWorkers.get(playerTurn);
						} else if (card.isBuilding) {
							cardDestination = playerBuildings.get(playerTurn);
						} else if (card.isAristocrat) {
							cardDestination = playerAristocrats.get(playerTurn);
						}
						// Reduce the cost by 1 for each same card in the destination
						for (SPCard c : cardDestination) {
							if (c == card) {
								cost--;
							}
						}
						// If the card is not a trading card, create a buy action if affordable
						if (!card.isTrading) {
							if (cost < 1) {
								cost = 1; // Ensure the cost is not less than 1
							}
							if (rubles >= cost) {
								// Create a buy action for the card
								SPBuyAction buyAction = new SPBuyAction(this, card, cardSource, cardDestination, cost);
								legalActions.add(buyAction);
							}
						} else { // If the card is a trading card, create a buy action(s) with replaced card(s)
							// Compute the replaceable cards with associated costs
							Map<SPCard, Integer> replaceableCards = new HashMap<>();
							for (SPCard c : cardDestination) {
								if (c.isTrading) {
									continue; // cannot replace trading cards with trading cards
								}
														// Worker trading cards have to replace compatible worker cards
								if (c.isWorker) {
									// determine if the trading card can replace the worker card
									if (!c.name.equals("Czar and Carpenter")) // automatically compatible
									{
										String cardNotes = card.other;
										// if the card notes has the string "replaces ",
										// then extract the trading card name thereafter to the end of the string
										int index = cardNotes.indexOf("replaces ");
										String replacedWorkerType = cardNotes.substring(index + 9).trim();
										if (!c.name.equalsIgnoreCase(replacedWorkerType)) {
											continue; // incompatible worker card
										}
									}
								}
								int replaceCost = cost - c.cost; // Cost of replacing the card
								if (c.name.equals("Observatory")) {
									if (numUnusedObservatories < 1) {
										continue; // Cannot replace an Observatory if none are left unused
									}
								}
								if (card.name.endsWith("Village")) {
									replaceCost -= 4; // Potjomkin's/Potemkin Village credits 6 while only costing 2
								}
								if (replaceCost < 1) {
									replaceCost = 1; // Ensure the cost is not less than 1
								}
								if (rubles >= replaceCost) {
									replaceableCards.put(c, replaceCost); // Add the card and its cost to the map
								}
							}
							// Create a buy action for each replaceable card
							for (Map.Entry<SPCard, Integer> entry : replaceableCards.entrySet()) {
								SPCard replacedCard = entry.getKey();
								int replaceCost = entry.getValue();
								SPBuyAction buyAction = new SPBuyAction(this, card, cardSource, cardDestination, replaceCost, replacedCard);
								legalActions.add(buyAction);
							}
						} // end trading card handling
					} // end for each card in card source
				} // end for each card source
			} // end buy actions

			// Add observe actions
			if (numUnusedObservatories > 0) { // Only add observe actions if the player has unused observatories
				if (workerDeck.size() > 1) {
					legalActions.add(new SPObserveDeckAction(this, 0));
				}
				if (buildingDeck.size() > 1) {
					legalActions.add(new SPObserveDeckAction(this, 1));
				}
				if (aristocratDeck.size() > 1) {
					legalActions.add(new SPObserveDeckAction(this, 2));
				}
				if (tradingDeck.size() > 1) {
					legalActions.add(new SPObserveDeckAction(this, 3));
				}
			} // end observe actions

			// Add add-to-hand actions

			if (hasRoomInHand) {
				// Add actions to add cards from the upper card row to the player's hand
				for (SPCard card : upperCardRow) {
					legalActions.add(new SPAddToHandAction(this, card, upperCardRow));
				}
				// Add actions to add cards from the lower card row to the player's hand
				for (SPCard card : lowerCardRow) {
					legalActions.add(new SPAddToHandAction(this, card, lowerCardRow));
				}
			}
		} // end else for normal actions
		return legalActions;
	}

	public static void main(String[] args) {
		// Test the SPState class by creating an instance and printing it
		SPState state = new SPState();
		System.out.println(state);
	}	
}
