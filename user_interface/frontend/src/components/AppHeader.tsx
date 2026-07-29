import { Link } from 'react-router-dom'
import { Sprout } from 'lucide-react'
import type { ReactNode } from 'react'

interface AppHeaderProps {
  action?: ReactNode
}

export function AppHeader({ action }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <Link className="app-brand" to="/" aria-label="FarmAI home">
          <span className="app-brand__mark">
            <Sprout size={20} aria-hidden="true" />
          </span>
          FarmAI
        </Link>
        {action}
      </div>
    </header>
  )
}
