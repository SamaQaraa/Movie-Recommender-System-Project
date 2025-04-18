import sys
import random
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QPushButton, QListWidget,
                             QLabel, QGroupBox, QSpinBox, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer

class UserMoviesApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadDummyData()
        
    def initUI(self):
        self.setWindowTitle('User Movies Recommendation System')
        self.setGeometry(100, 100, 800, 500)
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create user selection area
        selection_group = QGroupBox("User Selection")
        selection_layout = QHBoxLayout()
        
        # Create user ID combobox
        self.user_combo = QComboBox()
        self.user_combo.setMinimumWidth(150)
        
        # Create show button
        self.show_button = QPushButton("Show User Movies")
        self.show_button.clicked.connect(self.showUserMovies)
        
        # Add widgets to selection layout
        selection_layout.addWidget(QLabel("Select User ID:"))
        selection_layout.addWidget(self.user_combo)
        selection_layout.addWidget(self.show_button)
        selection_layout.addStretch()
        selection_group.setLayout(selection_layout)
        
        # Create recommendation controls
        recommendation_group = QGroupBox("Recommendation Controls")
        recommendation_layout = QHBoxLayout()
        
        # Create number of recommendations spinbox
        recommendation_layout.addWidget(QLabel("Number of recommendations:"))
        self.num_recommendations = QSpinBox()
        self.num_recommendations.setMinimum(1)
        self.num_recommendations.setMaximum(100)
        self.num_recommendations.setValue(5)
        recommendation_layout.addWidget(self.num_recommendations)
        
        # Create get recommendations button
        self.recommend_button = QPushButton("Get Movie Recommendations")
        self.recommend_button.clicked.connect(self.getRecommendations)
        recommendation_layout.addWidget(self.recommend_button)
        
        recommendation_layout.addStretch()
        recommendation_group.setLayout(recommendation_layout)
        
        # Create movies display area
        display_group = QGroupBox("Movies")
        display_layout = QHBoxLayout()
        
        # Create user movies area
        user_movies_layout = QVBoxLayout()
        user_movies_layout.addWidget(QLabel("User's Favorite Movies:"))
        self.user_movies_list = QListWidget()
        user_movies_layout.addWidget(self.user_movies_list)
        
        # Create recommended movies area
        recommended_layout = QVBoxLayout()
        recommended_layout.addWidget(QLabel("Recommended Movies:"))
        self.recommended_list = QListWidget()
        recommended_layout.addWidget(self.recommended_list)
        
        # Add both areas to display layout
        display_layout.addLayout(user_movies_layout)
        display_layout.addLayout(recommended_layout)
        display_group.setLayout(display_layout)
        
        # Add all components to main layout
        main_layout.addWidget(selection_group)
        main_layout.addWidget(recommendation_group)
        main_layout.addWidget(display_group)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
    def loadDummyData(self):
        # Create dummy data for users and their favorite movies
        self.user_data = {
            "user001": [
                "The Shawshank Redemption",
                "The Godfather",
                "The Dark Knight",
                "Pulp Fiction",
                "Fight Club"
            ],
            "user002": [
                "Inception",
                "Interstellar",
                "The Matrix",
                "Blade Runner 2049",
                "Arrival"
            ],
            "user003": [
                "The Lord of the Rings: The Fellowship of the Ring",
                "Star Wars: Episode V - The Empire Strikes Back",
                "Harry Potter and the Prisoner of Azkaban",
                "The Hobbit: An Unexpected Journey",
                "Dune"
            ],
            "user004": [
                "Titanic",
                "The Notebook",
                "Pride and Prejudice",
                "La La Land",
                "A Star Is Born"
            ],
            "user005": [
                "The Avengers",
                "Iron Man",
                "Black Panther",
                "Spider-Man: Into the Spider-Verse",
                "The Dark Knight Rises"
            ]
        }
        
        # Create dummy recommendation database
        self.recommendation_db = {
            "user001": [
                "The Godfather: Part II",
                "Goodfellas",
                "The Departed",
                "Se7en",
                "The Silence of the Lambs",
                "American History X",
                "Memento",
                "The Usual Suspects",
                "Léon: The Professional",
                "No Country for Old Men"
            ],
            "user002": [
                "2001: A Space Odyssey",
                "The Martian",
                "Ex Machina",
                "Eternal Sunshine of the Spotless Mind",
                "Solaris",
                "Moon",
                "District 9",
                "Her",
                "WALL-E",
                "Primer"
            ],
            "user003": [
                "The Lord of the Rings: The Two Towers",
                "The Lord of the Rings: The Return of the King",
                "Star Wars: Episode IV - A New Hope",
                "The Princess Bride",
                "Narnia: The Lion, the Witch and the Wardrobe",
                "Stardust",
                "Pan's Labyrinth",
                "Avatar",
                "Harry Potter and the Goblet of Fire",
                "Game of Thrones"
            ],
            "user004": [
                "Romeo + Juliet",
                "When Harry Met Sally",
                "Sleepless in Seattle",
                "Before Sunrise",
                "The Fault in Our Stars",
                "Eternal Sunshine of the Spotless Mind",
                "500 Days of Summer",
                "Silver Linings Playbook",
                "About Time",
                "The Shape of Water"
            ],
            "user005": [
                "Thor: Ragnarok",
                "Captain America: The Winter Soldier",
                "Guardians of the Galaxy",
                "Logan",
                "Deadpool",
                "Wonder Woman",
                "The Dark Knight",
                "Shazam!",
                "Doctor Strange",
                "Watchmen"
            ]
        }
        
        # Populate user combobox
        for user_id in self.user_data.keys():
            self.user_combo.addItem(user_id)
    
    def showUserMovies(self):
        # Get selected user ID
        selected_user = self.user_combo.currentText()
        
        # Display user's favorite movies
        user_movies = self.user_data.get(selected_user, [])
        self.user_movies_list.clear()
        for movie in user_movies:
            item = QListWidgetItem(movie)
            self.user_movies_list.addItem(item)
        
        # Clear recommended movies area
        self.recommended_list.clear()
    
    def getRecommendations(self):
        # Get selected user ID and number of recommendations
        selected_user = self.user_combo.currentText()
        num_movies = self.num_recommendations.value()
        
        # Show loading message
        self.recommended_list.clear()
        loading_item = QListWidgetItem("Connecting to recommendation server...")
        self.recommended_list.addItem(loading_item)
        processing_item = QListWidgetItem("Processing request...")
        self.recommended_list.addItem(processing_item)
        
        # Simulate server request delay
        QTimer.singleShot(1500, lambda: self.displayRecommendations(selected_user, num_movies))
    
    def displayRecommendations(self, user_id, num_movies):
        # In a real application, this would make an API call to a recommendation engine
        # For now, we'll use our dummy database
        recommended_movies = self.recommendation_db.get(user_id, [])
        
        # Clear list widget
        self.recommended_list.clear()
        
        if not recommended_movies:
            self.recommended_list.addItem(QListWidgetItem("No recommendations available for this user."))
            return
        
        # Ensure we don't request more movies than available
        num_to_show = min(num_movies, len(recommended_movies))
        
        # Display recommendations
        header_item = QListWidgetItem(f"Top {num_to_show} Recommended Movies for {user_id}:")
        header_item.setFlags(Qt.ItemIsEnabled)  # Make it non-selectable
        self.recommended_list.addItem(header_item)
        
        for i in range(num_to_show):
            movie_item = QListWidgetItem(f"{i+1}. {recommended_movies[i]}")
            self.recommended_list.addItem(movie_item)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = UserMoviesApp()
    ex.show()
    sys.exit(app.exec_())