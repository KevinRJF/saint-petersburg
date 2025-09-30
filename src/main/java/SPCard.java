/* SPCard - an immutable class for representing cards in the boardgame Saint Petersburg.
 * This class loads card information from cards.csv and initializes immutable card lists for building the game decks.
 */

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class SPCard {
	// Immutable lists for each type of card
	public static final int EDITION = 1; // Edition number for the game
	public static final List<SPCard> WORKER_CARDS;
	public static final List<SPCard> BUILDING_CARDS;
	public static final List<SPCard> ARISTOCRAT_CARDS;
	public static final List<SPCard> TRADING_CARDS;
	public static final List<SPCard> ALL_CARDS;
	public static final int NUM_WORKER_CARDS, NUM_BUILDING_CARDS, NUM_ARISTOCRAT_CARDS, NUM_TRADING_CARDS, NUM_ALL_CARDS;

	public final int edition, quantity, cost, rubles, points;
	public final String type, name, other, abbrev;
	public final boolean isWorker, isBuilding, isAristocrat, isTrading;

	// Constructor to initialize card properties
	public SPCard(int edition, int quantity, String type, String name, int cost, int rubles, int points, String other,
			String abbrev) {
		this.edition = edition;
		this.quantity = quantity;
		this.type = type;
		this.name = name;
		this.cost = cost;
		this.rubles = rubles;
		this.points = points;
		this.other = other;
		this.abbrev = abbrev;

		// Determine card type
		isWorker = type.contains("worker");
		isAristocrat = type.contains("aristocrat");
		isBuilding = type.contains("building");
		isTrading = type.contains("trading");
	}

	static {
		ArrayList<SPCard> workerCards = new ArrayList<>();
		ArrayList<SPCard> buildingCards = new ArrayList<>();
		ArrayList<SPCard> aristocratCards = new ArrayList<>();
		ArrayList<SPCard> tradingCards = new ArrayList<>();
		ArrayList<SPCard> allCards = new ArrayList<>();

		// Load card data from CSV file
		try (Scanner scanner = new Scanner(new File("cards.csv"));) {
			
			scanner.nextLine(); // Skip header line
			while (scanner.hasNextLine()) {
				String line = scanner.nextLine();
				String[] values = line.split(",");
				int edition = Integer.parseInt(values[0].trim());
				if (edition != EDITION) {
					continue; // Skip cards not from the current edition
				}
				int quantity = Integer.parseInt(values[1].trim());
				String type = values[2].trim();
				String name = values[3].trim();
				int cost = Integer.parseInt(values[4].trim());
				int rubles = Integer.parseInt(values[5].trim());
				int points = Integer.parseInt(values[6].trim());
				String other = values[7].trim();
				String abbrev = values[8].trim();

				SPCard card = new SPCard(edition, quantity, type, name, cost, rubles, points, other, abbrev);
				allCards.add(card); // Add to all cards list

				// Add card to the appropriate list based on its type
				if (card.isTrading) {
					tradingCards.add(card);
				} else if (card.isBuilding) {
					buildingCards.add(card);
				} else if (card.isAristocrat) {
					aristocratCards.add(card);
				} else if (card.isWorker) {
					workerCards.add(card);
				} else {
					throw new IllegalArgumentException("Unknown card type: " + type);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		// Create unmodifiable lists for each type of card
		WORKER_CARDS = Collections.unmodifiableList(workerCards);
		ARISTOCRAT_CARDS = Collections.unmodifiableList(aristocratCards);
		BUILDING_CARDS = Collections.unmodifiableList(buildingCards);
		TRADING_CARDS = Collections.unmodifiableList(tradingCards);
		ALL_CARDS = Collections.unmodifiableList(allCards);

		// Calculate the number of cards in each category
		NUM_WORKER_CARDS = WORKER_CARDS.size();
		NUM_BUILDING_CARDS = BUILDING_CARDS.size();
		NUM_ARISTOCRAT_CARDS = ARISTOCRAT_CARDS.size();
		NUM_TRADING_CARDS = TRADING_CARDS.size();
		NUM_ALL_CARDS = ALL_CARDS.size();
	} // static block

	@Override
	public String toString() {
		return String.format("%s (%s)", name, abbrev);
	}

	public static void main(String[] args) {
		// Test the SPCard class by printing
		System.out.printf("Loaded %d worker cards, %d aristocrat cards, %d building cards, and %d trading cards.%n",
				WORKER_CARDS.size(), ARISTOCRAT_CARDS.size(), BUILDING_CARDS.size(), TRADING_CARDS.size());
		System.out.println("\nWorker Cards:");
		for (SPCard card : WORKER_CARDS) {
			System.out.println(card);
		}
		System.out.println("\nAristocrat Cards:");
		for (SPCard card : ARISTOCRAT_CARDS) {
			System.out.println(card);
		}
		System.out.println("\nBuilding Cards:");
		for (SPCard card : BUILDING_CARDS) {
			System.out.println(card);
		}
		System.out.println("\nTrading Cards:");
		for (SPCard card : TRADING_CARDS) {
			System.out.println(card);
		}
	}
}
