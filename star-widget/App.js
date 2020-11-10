import React from 'react';
import logo from './logo.svg';
import './App.css';
import StarRating from './StarRating';

function App() {
  return (
    <div className="App">
      <StarRating />
    </div>
  );
}

export default App;

const domContainer = document.querySelector('#star-widget-container');
ReactDOM.render(e(StarRating), domContainer);
