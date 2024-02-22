import React, { useState, useEffect } from 'react';
import axios from 'axios';

function GetGames() {
  const [message, setMessage] = useState('');

  useEffect(() => {
    axios.get('http://localhost:8000/api/get-recs')
      .then(response => {
        setMessage(response.data.message);
      })
      .catch(error => {
        console.log(error);
      });
  }, []);

  return (
    <div>
      <h1>Hello User!</h1>
      <p>{message}</p>
    </div>
  );
}

export default GetGames;