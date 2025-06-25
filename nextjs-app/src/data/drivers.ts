export interface Driver {
  id: number;            // driverId oficial (dataset Ergast)
  constructorId: number; // equipo actual
  name: string;
}

/**
 *  Lista mínima para compilar; completa según necesites.
 *  Los constructorId son los vigentes en 2025.
 */
export const drivers: Driver[] = [
  { id: 1,   constructorId: 131, name: "Lewis Hamilton" },
  { id: 822, constructorId: 213, name: "Max Verstappen" },
  { id: 815, constructorId: 6,   name: "Fernando Alonso" },
  { id: 830, constructorId: 9,   name: "Charles Leclerc" },
  { id: 832, constructorId: 117, name: "Lando Norris" },
  // …añade el resto de pilotos que utilices
];
