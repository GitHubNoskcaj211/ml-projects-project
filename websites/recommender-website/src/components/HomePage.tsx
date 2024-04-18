import React from 'react';
const steamIcon = 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Steam_icon_logo.svg/2048px-Steam_icon_logo.svg.png'; // URL for the steam icon image

const HomePage: React.FC = () => {
    return (
        <div className="bg-gray-900 text-white min-h-screen p-4 md:p-6 flex flex-col items-center justify-center">
            <div className="text-center mb-8">
                <img src={steamIcon} alt="Steam" className="w-16 h-16 md:w-20 md:h-20 mx-auto mb-4" />
                
                {/* Responsive text size */}
                <h1 className="text-3xl sm:text-3xl md:text-4xl lg:text-5xl xl:text-6xl font-bold mb-3">Steam Game Recommendation Engine</h1>
                <p className="text-sm md:text-xl">
                    Discover games that match your interests and dive into new worlds. Our
                    mission is to connect you with your next gaming adventure through
                    personalized recommendations and community-driven insights.
                </p>
            </div>

            <div className="flex flex-col md:flex-row gap-4 md:gap-6 justify-center items-stretch max-w-6xl w-full">
                {/* Box 1 */}
                <div className="flex-1 bg-gray-800 p-4 md:p-6 rounded-lg shadow-lg">
                    <h2 className="text-lg md:text-2xl font-semibold mb-2 md:mb-4">What We Do</h2>
                    <p className="text-xs md:text-sm">
                        Our platform offers personalized game recommendations based on your playing habits and preferences.
                        Explore new and trending titles or rediscover classics that align with your tastes.
                    </p>
                </div>
                {/* Box 2 */}
                <div className="flex-1 bg-gray-800 p-4 md:p-6 rounded-lg shadow-lg">
                    <h2 className="text-lg md:text-2xl font-semibold mb-2 md:mb-4">Get Recommended Games</h2>
                    <p className="text-xs md:text-sm">
                        Navigate to the <b>Find New Games</b> section to browse our extensive database of games.
                        Our recommendation system will suggest games that you're likely to enjoy based on your past selections.
                    </p>
                </div>
                {/* Box 3 */}
                <div className="flex-1 bg-gray-800 p-4 md:p-6 rounded-lg shadow-lg">
                    <h2 className="text-lg md:text-2xl font-semibold mb-2 md:mb-4">View Liked Games</h2>
                    <p className="text-xs md:text-sm">
                        In the <b>Liked Games</b> section, you can review the games you've marked as liked and keep track of your favorites.
                        This list helps us tailor your recommendations even further.
                    </p>
                </div>
            </div>
        </div>
    );
};

export default HomePage;
