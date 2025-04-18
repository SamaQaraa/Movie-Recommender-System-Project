# client.py
"""
PyQt5-based client interface for the movie recommendation system.
This client provides a graphical user interface to interact with the recommendation API.
"""

import sys
import json
import requests
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QPushButton, QListWidget,
                             QLabel, QGroupBox, QSpinBox, QListWidgetItem, 
                             QMessageBox)
from PyQt5.QtCore import Qt, QTimer

# Server API URL
API_URL = "http://127.0.0.1:8000"

class MovieRecommendationClient(QMainWindow):
    """Main window class for the movie recommendation client"""
    
    def __init__(self):
        """Initialize the client window and UI components"""
        super().__init__()
        self.initUI()
        self.loadUserIds()
        
    def initUI(self):
        """Initialize the user interface components"""
        self.setWindowTitle('Movie Recommendation Client')
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
        self.show_button.clicked.connect(self.fetchUserMovies)
        
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
        self.recommend_button.clicked.connect(self.fetchRecommendations)
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
    
    def loadUserIds(self):
        """Fetch available user IDs from server"""
        try:
            response = requests.get(f"{API_URL}/users")
            if response.status_code == 200:
                users = response.json().get("users", [])
                self.user_combo.clear()
                for user_id in users:
                    self.user_combo.addItem(user_id)
            else:
                self.showError(f"Failed to load users: {response.status_code}")
        except requests.exceptions.RequestException as e:
            self.showError(f"Connection error: {str(e)}")
    
    def fetchUserMovies(self):
        """Fetch favorite movies for selected user from server"""
        selected_user = self.user_combo.currentText()
        if not selected_user:
            return
            
        # Clear the list and show loading
        self.user_movies_list.clear()
        self.user_movies_list.addItem("Loading...")
        
        try:
            # Send request to server
            response = requests.get(f"{API_URL}/user/{selected_user}/movies")
            
            # Clear loading message
            self.user_movies_list.clear()
            
            if response.status_code == 200:
                movies = response.json().get("movies", [])
                if movies:
                    for movie in movies:
                        self.user_movies_list.addItem(movie)
                else:
                    self.user_movies_list.addItem("No favorite movies found for this user")
            else:
                self.showError(f"Failed to load movies: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.user_movies_list.clear()
            self.showError(f"Connection error: {str(e)}")
    
    def fetchRecommendations(self):
        """Fetch movie recommendations from server"""
        selected_user = self.user_combo.currentText()
        num_movies = self.num_recommendations.value()
        
        if not selected_user:
            return
            
        # Clear list and show loading
        self.recommended_list.clear()
        self.recommended_list.addItem("Connecting to recommendation server...")
        
        try:
            # Prepare request data
            request_data = {
                "user_id": selected_user,
                "count": num_movies
            }
            
            # Send request to server
            response = requests.post(
                f"{API_URL}/recommendations", 
                json=request_data
            )
            
            # Clear loading message
            self.recommended_list.clear()
            
            if response.status_code == 200:
                recommendations = response.json().get("movies", [])
                
                if recommendations:
                    header_item = QListWidgetItem(f"Top {len(recommendations)} Recommended Movies:")
                    header_item.setFlags(Qt.ItemIsEnabled)  # Make it non-selectable
                    self.recommended_list.addItem(header_item)
                    
                    for i, movie in enumerate(recommendations):
                        movie_item = QListWidgetItem(f"{i+1}. {movie}")
                        self.recommended_list.addItem(movie_item)
                else:
                    self.recommended_list.addItem("No recommendations available")
            else:
                error_message = response.json().get("detail", f"Server error: {response.status_code}")
                self.showError(error_message)
                
        except requests.exceptions.RequestException as e:
            self.recommended_list.clear()
            self.showError(f"Connection error: {str(e)}")
    
    def showError(self, message):
        """Show error message box"""
        QMessageBox.critical(self, "Error", message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = MovieRecommendationClient()
    client.show()
    sys.exit(app.exec_())
